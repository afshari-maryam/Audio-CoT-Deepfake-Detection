"""Evaluation script for COLMBO-DF (Section 4 & 5).

Metrics: Accuracy and F1 for both tasks:
  ADD (audio deepfake detection) – is Audio 2 real or fake?
  ASV (automatic speaker verification) – same or different speaker?

Invalid / unparseable outputs are treated as abstentions (wrong prediction).

Usage
-----
    python evaluate.py \
        --checkpoint  checkpoints/shortcot/checkpoint-5000 \
        --manifest    data/fakereason_eval.json \
        --mode        shortcot \
        --output      results/eval_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FAKEREASONDataset, collate_fn, _load_waveform
from features import extract_features, serialize_features
from inference import parse_prediction, build_inference_prompt
from model import ColmboDF


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, float]:
    """Compute accuracy and macro-F1 for ADD and ASV tasks."""
    from sklearn.metrics import accuracy_score, f1_score

    add_pred, add_true = [], []
    asv_pred, asv_true = [], []

    for pred, gt in zip(predictions, ground_truth):
        # ADD: speaker2 genuine→0, deepfake→1
        p_add = 1 if pred.get("speaker2", "unknown") == "deepfake" else 0
        t_add = 1 if gt["label_add"] == "fake" else 0
        add_pred.append(p_add)
        add_true.append(t_add)

        # ASV: same speaker→0, different→1
        raw = pred.get("speaker_relation", "unknown").lower()
        p_asv = 1 if "different" in raw else 0
        t_asv = 1 if gt["label_asv"] == "different" else 0
        asv_pred.append(p_asv)
        asv_true.append(t_asv)

    return {
        "acc_add": accuracy_score(add_true, add_pred),
        "f1_add":  f1_score(add_true, add_pred, average="macro", zero_division=0),
        "acc_asv": accuracy_score(asv_true, asv_pred),
        "f1_asv":  f1_score(asv_true, asv_pred, average="macro", zero_division=0),
    }


# ── Per-sample evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(
    model: ColmboDF,
    manifest_path: str,
    device: torch.device,
    max_audio_len: int = 80000,
    max_new_tokens: int = 256,
    batch_size: int = 1,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Run inference over all pairs in a manifest and return predictions + metrics."""
    with open(manifest_path) as f:
        items = json.load(f)

    predictions = []
    model.eval()

    for item in tqdm(items, desc="Evaluating"):
        wav1 = _load_waveform(item["audio1"], max_len=max_audio_len).unsqueeze(0).to(device)
        wav2 = _load_waveform(item["audio2"], max_len=max_audio_len).unsqueeze(0).to(device)

        input_ids = build_inference_prompt(
            item["audio1"],
            item["audio2"],
            model.tokenizer,
            features1=item.get("features1"),
            features2=item.get("features2"),
        ).to(device)

        try:
            output_ids = model.generate(
                waveform1=wav1,
                waveform2=wav2,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            parsed = parse_prediction(text)
        except Exception as e:
            print(f"  Error on {item['audio2']}: {e}")
            parsed = {"speaker2": "unknown", "speaker_relation": "unknown", "reasoning": ""}

        predictions.append({"raw": text if "text" in dir() else "", **parsed})

    metrics = compute_metrics(predictions, items)
    return predictions, metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate COLMBO-DF on a manifest")
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--manifest",         required=True)
    parser.add_argument("--encoder_name",     default="microsoft/wavlm-base-plus")
    parser.add_argument("--llm_name",         default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--max_audio_len",    type=int, default=80000)
    parser.add_argument("--max_new_tokens",   type=int, default=256)
    parser.add_argument("--output",           default="eval_results.json")
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

    predictions, metrics = evaluate_dataset(
        model, args.manifest, device,
        max_audio_len=args.max_audio_len,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out = {"metrics": metrics, "predictions": predictions}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
