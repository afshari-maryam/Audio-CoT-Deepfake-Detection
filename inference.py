"""Inference script for COLMBO-DF.

Loads a trained checkpoint and runs the model on an audio pair,
producing a chain-of-thought explanation and a final prediction.

Usage
-----
    python inference.py \
        --checkpoint  checkpoints/shortcot/checkpoint-5000 \
        --audio1      /path/to/real.wav \
        --audio2      /path/to/unknown.wav \
        --mode        shortcot
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchaudio

from features import extract_features, serialize_features
from model import ColmboDF, AUDIO_PLACEHOLDER
from dataset import USER_PROMPT_TEMPLATE


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_inference_prompt(
    audio1_path: str,
    audio2_path: str,
    tokenizer,
    features1: Optional[Dict] = None,
    features2: Optional[Dict] = None,
) -> torch.Tensor:
    """Build the user prompt (no assistant response) for inference."""
    if features1 is None:
        features1 = extract_features(audio1_path)
    if features2 is None:
        features2 = extract_features(audio2_path)

    feat1 = serialize_features(features1)
    feat2 = serialize_features(features2)
    user_content = USER_PROMPT_TEMPLATE.format(feat1=feat1, feat2=feat2)

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    return input_ids  # (1, L)


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_waveform(
    path: str, target_sr: int = 16000, max_len: int = 80000
) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if wav.shape[0] >= max_len:
        wav = wav[:max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[0]))
    return wav


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_prediction(text: str) -> Dict[str, str]:
    """Extract structured fields from the model's generated text."""
    import re

    result = {"speaker2": "unknown", "speaker_relation": "unknown", "reasoning": ""}

    m = re.search(r"- Speaker 2:\s*(Genuine|Deepfake)", text, re.IGNORECASE)
    if m:
        result["speaker2"] = m.group(1).strip().lower()

    m = re.search(
        r"- Speaker Relationship:\s*(Same Speaker|Different Speakers)",
        text, re.IGNORECASE
    )
    if m:
        result["speaker_relation"] = m.group(1).strip().lower()

    m = re.search(r"- Reasoning:\s*(.+?)(?:\n-|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()

    return result


# ── Main inference function ───────────────────────────────────────────────────

def predict(
    model: ColmboDF,
    audio1_path: str,
    audio2_path: str,
    device: torch.device,
    max_audio_len: int = 80000,
    max_new_tokens: int = 512,
    features1: Optional[Dict] = None,
    features2: Optional[Dict] = None,
) -> Tuple[str, Dict[str, str]]:
    """Run COLMBO-DF on a single audio pair.

    Returns
    -------
    generated_text : the full model output string
    parsed         : dict with keys speaker2, speaker_relation, reasoning
    """
    wav1 = load_waveform(audio1_path, max_len=max_audio_len).unsqueeze(0).to(device)
    wav2 = load_waveform(audio2_path, max_len=max_audio_len).unsqueeze(0).to(device)

    input_ids = build_inference_prompt(
        audio1_path, audio2_path, model.tokenizer, features1, features2
    ).to(device)

    model.eval()
    output_ids = model.generate(
        waveform1=wav1,
        waveform2=wav2,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    parsed = parse_prediction(generated)
    return generated, parsed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COLMBO-DF inference on an audio pair")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--audio1",      required=True,
                        help="Path to genuine reference audio (.wav)")
    parser.add_argument("--audio2",      required=True,
                        help="Path to unknown target audio (.wav)")
    parser.add_argument("--encoder_name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--llm_name",     default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens",   type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model …")
    model = ColmboDF(
        encoder_name=args.encoder_name,
        llm_name=args.llm_name,
        num_query_tokens=args.num_query_tokens,
        freeze_encoder=True,
        freeze_llm=True,
    ).to(device)

    ckpt = torch.load(Path(args.checkpoint) / "state.pt", map_location="cpu")
    model.projector.load_state_dict(ckpt["projector"])
    if ckpt.get("llm") is not None:
        model.llm.load_state_dict(ckpt["llm"])
    print("Checkpoint loaded.")

    print(f"\nAnalysing:\n  Audio 1 (ref): {args.audio1}\n  Audio 2 (unk): {args.audio2}\n")
    text, parsed = predict(
        model, args.audio1, args.audio2, device,
        max_new_tokens=args.max_new_tokens,
    )

    print("=== Generated output ===")
    print(text)
    print("\n=== Parsed prediction ===")
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
