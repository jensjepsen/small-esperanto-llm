"""LlamaConfig factory and TrainingArguments defaults loaded from YAML configs."""

import os
from pathlib import Path

import yaml
from transformers import LlamaConfig, TrainingArguments

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_yaml_config(config_name: str) -> dict:
    path = CONFIGS_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def make_llama_config(config_name: str) -> LlamaConfig:
    cfg = load_yaml_config(config_name)
    model_cfg = cfg["model"]
    return LlamaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg["num_key_value_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg["rms_norm_eps"],
    )


def make_training_args(config_name: str, output_dir: str) -> TrainingArguments:
    cfg = load_yaml_config(config_name)
    t = cfg["training"]

    import torch
    optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    # Auto-detect bf16 support (Ampere+); fall back to fp16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
    else:
        use_bf16 = False
        use_fp16 = t["fp16"]

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=t["max_grad_norm"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        dataloader_num_workers=t["dataloader_num_workers"],
        dataloader_pin_memory=t["dataloader_pin_memory"],
        optim=optim,
    )
