"""Data preparation: build the FAKEREASON manifest from ASVspoof 2019 + CosyFish.

Steps
-----
1. Walk the ASVspoof 2019 LA directory and read the official protocol files
   to determine genuine/spoofed labels and speaker IDs.
2. (Optional) Walk CosyFish directories for synthesised audio pairs from
   VoxCeleb2 + Fish-Speech / CosyVoice2.
3. Pair each spoofed (or genuine) utterance with a genuine reference from the
   same or a different speaker according to the desired ratio.
4. Extract acoustic features for every file.
5. Write a JSON manifest ready for generate_cot.py.

Usage
-----
    # ASVspoof 2019 only (repurpose eval as extra train data – Section 3):
    python prepare_data.py \
        --asvspoof_root  /data/ASVspoof2019/LA \
        --output_train   data/fakereason_train.json \
        --output_eval    data/fakereason_eval.json

    # With CosyFish:
    python prepare_data.py \
        --asvspoof_root  /data/ASVspoof2019/LA \
        --cosyfish_root  /data/CosyFish \
        --output_train   data/fakereason_train.json \
        --output_eval    data/fakereason_eval.json
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from features import extract_features

CHECKPOINT_DIR_DEFAULT = "/content/drive/MyDrive/COLMBO-DF-checkpoints"


# ── ASVspoof 2019 LA protocol parser ─────────────────────────────────────────

def parse_asv_protocol(protocol_file: Path) -> List[Dict]:
    """Parse an ASVspoof2019 LA protocol file.

    Expected columns (space-separated):
        speaker_id  file_id  <unused>  attack_type  label(bonafide/spoof)
    """
    entries = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            entries.append({
                "speaker_id":  parts[0],
                "file_id":     parts[1],
                "attack_type": parts[3],
                "label":       parts[4],  # "bonafide" | "spoof"
            })
    return entries


def build_asv_splits(
    root: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Return (train_entries, eval_entries) from ASVspoof2019 LA.

    Paper design (Section 3):
      Train = original ASVspoof train split  +  original ASVspoof eval split
      Eval  = original ASVspoof dev split
    """
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"

    train_proto = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto   = protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto  = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_entries = parse_asv_protocol(train_proto)
    dev_entries   = parse_asv_protocol(dev_proto)
    eval_entries  = parse_asv_protocol(eval_proto)

    # Attach full paths
    def attach_path(entries, split_name):
        flac_dir = root / f"ASVspoof2019_LA_{split_name}" / "flac"
        for e in entries:
            e["path"] = str(flac_dir / f"{e['file_id']}.flac")
        return entries

    train_entries = attach_path(train_entries, "train")
    dev_entries   = attach_path(dev_entries,   "dev")
    eval_entries  = attach_path(eval_entries,  "eval")

    # Merge train + original-eval as training data
    all_train = train_entries + eval_entries
    return all_train, dev_entries


# ── Pair construction ────────────────────────────────────────────────────────

def build_pairs(
    entries: List[Dict],
    same_spkr_ratio: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """Construct (reference, target) pairs as described in Section 3.

    Each target is paired with:
      - a genuine utterance from the SAME speaker   (prob = same_spkr_ratio)
      - a genuine utterance from a DIFFERENT speaker (prob = 1 - same_spkr_ratio)

    The reference is always genuine.
    """
    rng = random.Random(seed)

    genuine_by_spkr: Dict[str, List[Dict]] = {}
    for e in entries:
        if e["label"] == "bonafide":
            genuine_by_spkr.setdefault(e["speaker_id"], []).append(e)

    all_genuine = [e for lst in genuine_by_spkr.values() for e in lst]
    pairs = []

    for target in entries:
        spkr = target["speaker_id"]
        same_pool = [g for g in genuine_by_spkr.get(spkr, []) if g["file_id"] != target["file_id"]]
        diff_pool = [g for g in all_genuine if g["speaker_id"] != spkr]

        if rng.random() < same_spkr_ratio and same_pool:
            ref = rng.choice(same_pool)
            asv_label = "same"
        elif diff_pool:
            ref = rng.choice(diff_pool)
            asv_label = "different"
        elif same_pool:
            ref = rng.choice(same_pool)
            asv_label = "same"
        else:
            continue  # skip if no reference available

        pairs.append({
            "audio1":    ref["path"],
            "audio2":    target["path"],
            "label_add": "real" if target["label"] == "bonafide" else "fake",
            "label_asv": asv_label,
        })

    return pairs


# ── CosyFish helpers ──────────────────────────────────────────────────────────

def build_cosyfish_pairs(cosyfish_root: Path, seed: int = 42) -> List[Dict]:
    """Walk a CosyFish directory (synthesised deepfakes + VoxCeleb2 bonafide).

    Expected directory layout:
        <cosyfish_root>/
            real/   *.wav   (VoxCeleb2 utterances)
            fake/   *.wav   (TTS-synthesised deepfakes)
    """
    real_files = sorted((cosyfish_root / "real").glob("*.wav"))
    fake_files = sorted((cosyfish_root / "fake").glob("*.wav"))

    rng = random.Random(seed)
    pairs = []

    for fake in fake_files:
        ref = rng.choice(real_files)
        pairs.append({
            "audio1":    str(ref),
            "audio2":    str(fake),
            "label_add": "fake",
            "label_asv": "different",  # TTS outputs have no ground-truth speaker ID
        })

    for real in real_files:
        others = [r for r in real_files if r != real]
        if others:
            ref = rng.choice(others)
            pairs.append({
                "audio1":    str(ref),
                "audio2":    str(real),
                "label_add": "real",
                "label_asv": "different",
            })

    return pairs


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_one(item: Dict) -> Dict:
    """Worker function: extract features for one pair (runs in subprocess)."""
    item["features1"] = extract_features(item["audio1"])
    item["features2"] = extract_features(item["audio2"])
    item["cot"]       = ""
    item["cot_short"] = ""
    return item


def add_features(
    pairs: List[Dict],
    checkpoint_path: Optional[Path] = None,
    save_every: int = 100,
    num_workers: int = 4,
) -> List[Dict]:
    """Extract features in parallel with incremental checkpointing."""
    done: List[Dict] = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done = json.load(f)
        print(f"  Resuming from checkpoint: {len(done)}/{len(pairs)} done")

    remaining = pairs[len(done):]
    if not remaining:
        return done

    pbar = tqdm(total=len(pairs), initial=len(done), desc="Extracting features")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_extract_one, item): i
                   for i, item in enumerate(remaining)}
        # Collect in completion order, then re-sort to keep original order
        results = [None] * len(remaining)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            pbar.update(1)
            if (len(done) + idx + 1) % save_every == 0 and checkpoint_path:
                # Save what we have so far (only fully-done items)
                partial = done + [r for r in results[:idx+1] if r is not None]
                with open(checkpoint_path, "w") as f:
                    json.dump(partial, f)

    pbar.close()
    done.extend(results)

    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump(done, f)

    return done


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build FAKEREASON manifest")
    parser.add_argument("--asvspoof_root",   required=True)
    parser.add_argument("--cosyfish_root",   default=None)
    parser.add_argument("--output_train",    default="data/fakereason_train.json")
    parser.add_argument("--output_eval",     default="data/fakereason_eval.json")
    parser.add_argument("--checkpoint_dir",  default=CHECKPOINT_DIR_DEFAULT,
                        help="Directory on Drive for incremental checkpoints")
    parser.add_argument("--save_every",      type=int, default=100,
                        help="Save checkpoint every N feature extractions")
    parser.add_argument("--num_workers",     type=int, default=4,
                        help="Parallel workers for feature extraction")
    parser.add_argument("--same_spkr_ratio", type=float, default=0.5)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    asvspoof_root = Path(args.asvspoof_root)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Final manifests go to the checkpoint dir (outside the git repo)
    out_train = ckpt_dir / "fakereason_train.json"
    out_eval  = ckpt_dir / "fakereason_eval.json"
    # Incremental checkpoint files (separate from finals)
    ckpt_train = ckpt_dir / "fakereason_train_ckpt.json"
    ckpt_eval  = ckpt_dir / "fakereason_eval_ckpt.json"

    # Also keep local symlinks in data/ so existing code doesn't need changes
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)

    print("Parsing ASVspoof 2019 LA …")
    train_entries, eval_entries = build_asv_splits(asvspoof_root)

    print("Building pairs …")
    train_pairs = build_pairs(train_entries, args.same_spkr_ratio, args.seed)
    eval_pairs  = build_pairs(eval_entries,  args.same_spkr_ratio, args.seed)

    if args.cosyfish_root:
        print("Adding CosyFish pairs …")
        cf_pairs = build_cosyfish_pairs(Path(args.cosyfish_root), args.seed)
        split_idx = int(0.8 * len(cf_pairs))
        train_pairs += cf_pairs[:split_idx]
        eval_pairs  += cf_pairs[split_idx:]

    print("Extracting acoustic features for train …")
    train_pairs = add_features(train_pairs, ckpt_train, args.save_every, args.num_workers)
    print("Extracting acoustic features for eval …")
    eval_pairs  = add_features(eval_pairs,  ckpt_eval,  args.save_every, args.num_workers)

    # Write final manifests to Drive
    with open(out_train, "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(out_eval, "w") as f:
        json.dump(eval_pairs, f, indent=2)

    # Write local copies so train.py / generate_cot.py find them without changes
    with open(args.output_train, "w") as f:
        json.dump(train_pairs, f, indent=2)
    with open(args.output_eval, "w") as f:
        json.dump(eval_pairs, f, indent=2)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Eval  pairs: {len(eval_pairs)}")
    print(f"Saved to Drive: {out_train}")
    print(f"Saved to Drive: {out_eval}")
    print("Next step: run generate_cot.py to add CoT annotations.")


if __name__ == "__main__":
    main()
