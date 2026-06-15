"""Acoustic feature extraction for COLMBO-DF.

Features extracted per audio sample:
  F0 (mean, std), silence ratio, jitter (local),
  energy (mean, std), spectral centroid (mean, std),
  formants F1/F2/F3 (mean).
"""

import warnings
from typing import Dict

import numpy as np
import librosa

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    _PRAAT_AVAILABLE = True
except ImportError:
    _PRAAT_AVAILABLE = False


def extract_features(audio_path: str, sr: int = 16000) -> Dict[str, float]:
    """Return a dict of scalar acoustic features for a single audio file."""
    y, _ = librosa.load(audio_path, sr=sr)
    feats: Dict[str, float] = {}

    # --- F0 / pitch ---
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
    feats["f0_mean"] = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    feats["f0_std"]  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else 0.0

    # --- Energy / breath ---
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    feats["energy_mean"] = float(np.mean(rms))
    feats["energy_std"]  = float(np.std(rms))

    # --- Silence ratio (fraction of frames below 20th-percentile energy) ---
    feats["silence_ratio"] = float(np.mean(rms < np.percentile(rms, 20)))

    # --- Spectral centroid ---
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_mean"] = float(np.mean(sc))
    feats["spectral_std"]  = float(np.std(sc))

    # --- Jitter and formants via Praat ---
    if _PRAAT_AVAILABLE:
        try:
            snd = parselmouth.Sound(audio_path)

            pp = praat_call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
            jitter = praat_call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            feats["jitter"] = 0.0 if (jitter is None or np.isnan(jitter)) else float(jitter)

            formant = praat_call(snd, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)
            for idx, name in enumerate(["f1", "f2", "f3"], start=1):
                val = praat_call(formant, "Get mean", idx, 0.0, 0.0, "Hertz")
                feats[f"{name}_mean"] = 0.0 if (val is None or np.isnan(val)) else float(val)
        except Exception:
            feats.update({"jitter": 0.0, "f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0})
    else:
        feats.update({"jitter": 0.0, "f1_mean": 0.0, "f2_mean": 0.0, "f3_mean": 0.0})

    return feats


def serialize_features(feats: Dict[str, float]) -> str:
    """Convert feature dict to the structured text string injected into the LLM prompt."""
    return (
        f"F0 mean/std: {feats.get('f0_mean', 0):.1f}/{feats.get('f0_std', 0):.1f} Hz; "
        f"Silence: {feats.get('silence_ratio', 0):.3f}; "
        f"Jitter: {feats.get('jitter', 0):.4f}; "
        f"Energy mean/std: {feats.get('energy_mean', 0):.5f}/{feats.get('energy_std', 0):.5f}; "
        f"Spectral centroid mean/std: {feats.get('spectral_mean', 0):.1f}/{feats.get('spectral_std', 0):.1f} Hz; "
        f"Formants F1/F2/F3: {feats.get('f1_mean', 0):.0f}/{feats.get('f2_mean', 0):.0f}/{feats.get('f3_mean', 0):.0f} Hz"
    )
