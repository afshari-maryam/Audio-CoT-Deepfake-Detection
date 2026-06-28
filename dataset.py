"""FAKEREASON dataset (Section 3).

Each sample is an audio pair:
  audio1  – known genuine reference speech
  audio2  – unknown target (genuine or deepfake)

The JSON manifest has entries of the form:
{
    "audio1":    "/abs/path/ref.wav",
    "audio2":    "/abs/path/target.wav",
    "label_add": "real" | "fake",
    "label_asv": "same" | "different",
    "features1": {<feature dict>},
    "features2": {<feature dict>},
    "cot":       "<full chain-of-thought text>",
    "cot_short": "<~100-word summary reasoning>"
}

Modes
-----
  "cot"      – full chain-of-thought supervision
  "shortcot" – only the short summary + structured conclusion
  "nocot"    – structured conclusion labels only (no reasoning text)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset

from features import serialize_features

CONCLUSION_TEMPLATE = (
    "- Speaker 1: Genuine\n"
    "- Speaker 2: {add_label}\n"
    "- Speaker Relationship: {asv_label}"
)

USER_PROMPT_TEMPLATE = (
    "<|audio|> <|audio|>\n"
    "Audio 1 is known real speech.\n"
    "Is Audio 2 real or fake? Are they the same speaker?\n"
    "Acoustic Evidence:\n"
    "[Audio 1] {feat1}\n"
    "[Audio 2] {feat2}"
)


def _build_response(item: Dict, mode: str) -> str:
    add_label = "Genuine" if item["label_add"] == "real" else "Deepfake"
    asv_label = "Same Speaker" if item["label_asv"] == "same" else "Different Speakers"
    conclusion = CONCLUSION_TEMPLATE.format(add_label=add_label, asv_label=asv_label)

    if mode == "cot":
        reasoning = item.get("cot", "").strip()
    elif mode == "shortcot":
        reasoning = item.get("cot_short", item.get("cot", "")).strip()
    else:
        reasoning = ""

    return (reasoning + "\n" + conclusion).strip() if reasoning else conclusion


def build_prompt(
    item: Dict,
    tokenizer,
    mode: str = "shortcot",
    max_length: int = 1024,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """Tokenize one (user, assistant) exchange and create loss labels.

    Labels are -100 for the user-prompt tokens (not supervised) and
    equal to input_ids for the assistant response tokens.
    The two <|audio|> placeholder tokens in the user prompt are also
    set to -100 in labels (they will be replaced by audio embeddings).
    """
    feat1 = serialize_features(item["features1"])
    feat2 = serialize_features(item["features2"])
    user_content = USER_PROMPT_TEMPLATE.format(feat1=feat1, feat2=feat2)
    assistant_content = _build_response(item, mode)

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize prompt only (to know where the response starts)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    prompt_len = len(prompt_ids)

    # Tokenize full sequence
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]

    # Build labels: mask prompt, keep response
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100

    return {"input_ids": input_ids, "labels": labels}


def _load_waveform(
    path: str, target_sr: int = 16000, max_len: int = 80000
) -> torch.Tensor:
    """Load, resample, mono-mix, and truncate/pad a waveform to max_len."""
    try:
        # soundfile handles flac reliably on Drive
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # (channels, T)
    except Exception:
        wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # (T,)
    if wav.shape[0] >= max_len:
        wav = wav[:max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[0]))
    return wav


class FAKEREASONDataset(Dataset):
    """Paired audio dataset with chain-of-thought annotations."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        mode: str = "shortcot",
        max_text_len: int = 1024,
        max_audio_len: int = 80000,
        sample_rate: int = 16000,
    ):
        with open(manifest_path) as f:
            self.items: List[Dict] = json.load(f)

        self.tokenizer = tokenizer
        self.mode = mode
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]

        wav1 = _load_waveform(item["audio1"], self.sample_rate, self.max_audio_len)
        wav2 = _load_waveform(item["audio2"], self.sample_rate, self.max_audio_len)

        text_fields = build_prompt(
            item,
            self.tokenizer,
            mode=self.mode,
            max_length=self.max_text_len,
            training=True,
        )

        return {
            "waveform1":  wav1,
            "waveform2":  wav2,
            "input_ids":  text_fields["input_ids"],
            "labels":     text_fields["labels"],
            "label_add":  item["label_add"],
            "label_asv":  item["label_asv"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    waveform1  = torch.stack([b["waveform1"]  for b in batch])
    waveform2  = torch.stack([b["waveform2"]  for b in batch])
    input_ids  = torch.stack([b["input_ids"]  for b in batch])
    labels     = torch.stack([b["labels"]     for b in batch])
    return {
        "waveform1": waveform1,
        "waveform2": waveform2,
        "input_ids": input_ids,
        "labels":    labels,
    }
