"""Chain-of-thought annotation generation (Section 3).

Uses the prompt template from Figure 3 to ask a stronger LLM (Qwen3-30B
or any OpenAI-compatible endpoint) to write the reasoning for each sample.

The generated text is stored back into the manifest JSON.

Usage
-----
    python generate_cot.py \
        --manifest  data/asvspoof_pairs.json \
        --output    data/fakereason_train.json \
        --model     Qwen/Qwen3-30B-A3B-Instruct \
        --short                          # also produce cot_short summaries
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

from config import DRIVE_ROOT

# ── Prompt templates (Figure 3) ──────────────────────────────────────────────

COT_SYSTEM_PROMPT = (
    "You are an expert in audio forensics and deepfake detection."
)

COT_USER_TEMPLATE = """\
Analyze the following pair of audio samples based on their acoustic features.
The first audio is the reference audio you are comparing against. The second \
audio is an unknown sample. Use the provided ground truth to keep your \
reasoning consistent with reality.

Audio 1:
  Label: {label1}
  Features: {feat1}

Audio 2:
  Label: {label2}
  Features: {feat2}

Ground Truth (for training reference, DO NOT contradict):
  - Audio 1 authenticity: {label1}
  - Audio 2 authenticity: {label2}
  - Speaker relationship: {asv_label}

Task: Produce a single, coherent reasoning narrative (no numbered steps) \
that covers:
  - Key acoustic traits of Audio 1 (pitch, formants, voice quality, prosody)
  - Key acoustic traits of Audio 2
  - Important similarities and differences between the two recordings
  - Assessment of the second audio's authenticity (genuine vs. deepfake) \
with evidence
  - Justification of whether the speakers are the same person or different \
people
  - An explicit acknowledgment that the reasoning aligns with the provided \
ground truth

Keep the reasoning faithful to the ground truth, while explaining how \
acoustic evidence supports it.
You MUST PRETEND LIKE you DO NOT KNOW the ground truth labels when analyzing \
the features, but your final conclusion MUST MATCH the ground truth.

Final Conclusion (use this exact structure at the end):
- Speaker 1: [Genuine / Deepfake]
- Speaker 2: [Genuine / Deepfake]
- Speaker Relationship: [Same Speaker / Different Speakers]
- Reasoning: [A shorter rephrasing of your reasoning process in natural \
language, around 100 words]
"""

SHORT_COT_SYSTEM = (
    "You are an expert in audio forensics. "
    "Summarize the following deepfake detection reasoning in ~100 words, "
    "keeping the same conclusion."
)


# ── HuggingFace model singleton (loaded once, reused for all samples) ─────────

_MODEL = None
_TOKENIZER = None


def _load_model(model_name: str, device: str):
    """Load model and tokenizer once and cache them globally."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading model {model_name} ...")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        print("Model loaded.")
    return _MODEL, _TOKENIZER


def _hf_generate(
    model_name: str,
    system: str,
    user: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    """Run generation reusing the already-loaded model."""
    import torch
    model, tokenizer = _load_model(model_name, device)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ── Per-sample CoT generation ─────────────────────────────────────────────────

def generate_cot_for_sample(
    item: Dict,
    model_name: str,
    device: str = "cpu",
    generate_short: bool = True,
) -> Dict:
    """Add `cot` (and optionally `cot_short`) fields to a manifest item."""
    from features import serialize_features

    feat1_str = serialize_features(item["features1"])
    feat2_str = serialize_features(item["features2"])

    label1 = "Genuine"
    label2 = "Genuine" if item["label_add"] == "real" else "Deepfake"
    asv_label = (
        "Same Speaker" if item["label_asv"] == "same" else "Different Speakers"
    )

    user_msg = COT_USER_TEMPLATE.format(
        label1=label1,
        feat1=feat1_str,
        label2=label2,
        feat2=feat2_str,
        asv_label=asv_label,
    )

    full_cot = _hf_generate(
        model_name, COT_SYSTEM_PROMPT, user_msg, device=device
    )

    # Strip any accidental "ground truth" references the model may have leaked
    full_cot = re.sub(
        r"(I know|given that|since the ground truth|the ground truth (is|says))[^.]*\.",
        "",
        full_cot,
        flags=re.IGNORECASE,
    ).strip()

    item["cot"] = full_cot

    if generate_short:
        # Extract the "Reasoning:" line from the structured conclusion as short CoT
        m = re.search(r"- Reasoning:\s*(.+)", full_cot, re.IGNORECASE | re.DOTALL)
        if m:
            item["cot_short"] = m.group(1).strip()
        else:
            # Fall back: summarise with the same model
            short = _hf_generate(
                model_name,
                SHORT_COT_SYSTEM,
                full_cot,
                max_new_tokens=200,
                device=device,
            )
            item["cot_short"] = short.strip()

    return item


# ── CLI entry-point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CoT annotations for FAKEREASON")
    parser.add_argument("--manifest",  default=f"{DRIVE_ROOT}/fakereason_train.json",
                        help="Input JSON manifest (no CoT yet)")
    parser.add_argument("--output",    default=f"{DRIVE_ROOT}/fakereason_train.json",
                        help="Output JSON manifest with CoT (overwrites in place)")
    parser.add_argument("--model",     default="Qwen/Qwen3-8B",
                        help="HuggingFace model id for CoT generation")
    parser.add_argument("--device",    default="cuda", help="Torch device")
    parser.add_argument("--short",     action="store_true", help="Also generate cot_short")
    parser.add_argument("--start",     type=int, default=0, help="Resume from item index")
    args = parser.parse_args()

    with open(args.manifest) as f:
        items = json.load(f)

    out_path = Path(args.output)

    # Resume: load existing output and merge with manifest
    # Always keep ALL items — only update cot/cot_short for processed ones
    if out_path.exists() and out_path != Path(args.manifest):
        with open(out_path) as f:
            existing = json.load(f)
        cot_map = {x["audio2"]: x for x in existing if x.get("cot", "").strip()}
    else:
        cot_map = {}

    # Apply already-generated CoT back to full manifest
    for item in items:
        if item["audio2"] in cot_map:
            item["cot"]       = cot_map[item["audio2"]].get("cot", "")
            item["cot_short"] = cot_map[item["audio2"]].get("cot_short", "")

    skip = sum(1 for x in items if x.get("cot", "").strip())
    print(f"Resuming: {skip}/{len(items)} items already have CoT.")

    for i, item in enumerate(items):
        if i < args.start or item.get("cot", "").strip():
            continue
        print(f"[{i+1}/{len(items)}] {item['audio2']}")
        try:
            item = generate_cot_for_sample(
                item, args.model, device=args.device, generate_short=args.short
            )
        except Exception as e:
            print(f"  ERROR: {e}")
        # Save ALL items every time — manifest never gets truncated
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)

    print(f"Done. Saved {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
