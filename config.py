from dataclasses import dataclass, field

# Root directory on Google Drive — all outputs go here, outside the git repo
DRIVE_ROOT = "/content/drive/MyDrive/COLMBO-DF-checkpoints"


@dataclass
class ModelConfig:
    encoder_name: str = "microsoft/wavlm-base-plus"
    llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_query_tokens: int = 32
    qformer_layers: int = 6
    qformer_heads: int = 8
    freeze_encoder: bool = True
    freeze_llm: bool = True


@dataclass
class TrainConfig:
    manifest_train: str = f"{DRIVE_ROOT}/fakereason_train.json"
    manifest_eval: str = f"{DRIVE_ROOT}/fakereason_eval.json"
    output_dir: str = f"{DRIVE_ROOT}/checkpoints"
    batch_size: int = 4
    grad_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_audio_len: int = 80000      # 5 s at 16 kHz
    max_text_len: int = 1024
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    save_steps: int = 500
    # "cot" | "shortcot" | "nocot"
    mode: str = "shortcot"
    use_cosyfish: bool = False
    fp16: bool = False
    bf16: bool = True
