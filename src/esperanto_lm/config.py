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
        # Share input embed + lm_head weight. Saves ~vocab*hidden params
        # (~12.7M on 1024-hidden, 12.4k-vocab). For sub-1B-param models,
        # quality is ~identical or slightly better (light regularization);
        # only worth untying at 7B+. Must be set at model-init time;
        # can't be flipped on an existing checkpoint.
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", False),
    )


def _resolve_dataloader_workers(yaml_value: int) -> int:
    """Same-flag semantics as data.num_proc(): respects ESPLLM_NUM_PROC.

    Home boxes export ESPLLM_NUM_PROC=4 to avoid >4-core crashes (see
    feedback_cpu_thread_limit memory). Cloud boxes leave it unset → we
    bump dataloader workers to 16 (prefetch parallelism stops scaling
    past ~16 in practice; no point going higher even on 64-core hosts).
    YAML value is the floor / explicit override when neither env nor
    auto rules apply cleanly.
    """
    v = os.environ.get("ESPLLM_NUM_PROC", "").strip().lower()
    if v in ("", "auto"):
        return 16
    try:
        return max(1, int(v))
    except ValueError:
        return yaml_value


def make_training_args(config_name: str, output_dir: str, hub_model_id: str | None = None) -> TrainingArguments:
    cfg = load_yaml_config(config_name)
    t = cfg["training"]

    import torch
    # `optim` from YAML if set, otherwise auto-pick.
    # Recommended values:
    #   adamw_torch_fused    — default, fp32 momenta, no extra deps
    #   paged_adamw_8bit     — bitsandbytes 8-bit Adam, ~75% smaller optim
    #                          state. Use when VRAM-constrained (e.g. >400M
    #                          params on 80GB). Needs `pip install bitsandbytes`.
    #   adafactor            — factored 2nd moment, ~50% smaller; slight LM
    #                          convergence penalty
    optim = t.get("optim")
    if not optim:
        optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    # Auto-detect bf16 support (Ampere+); fall back to fp16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
    else:
        use_bf16 = False
        use_fp16 = t["fp16"]

    eval_strategy = t.get("eval_strategy", "steps")
    save_strategy = t.get("save_strategy", "steps")
    # HF's load_best_model_at_end requires eval_strategy == save_strategy
    # and both != "no". Turn it off in smoke/skip modes.
    load_best = eval_strategy != "no" and save_strategy != "no"

    return TrainingArguments(
        output_dir=output_dir,
        # max_steps overrides num_train_epochs if set. Smoke configs use max_steps.
        num_train_epochs=t.get("num_train_epochs", 1.0),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        # torch.compile speeds up Llama by 10-20% on fixed-shape pretrain
        # chunks. Adds 1-2 min startup compile. May interact with Liger
        # kernels — test with a short run before committing to a long one.
        torch_compile=t.get("torch_compile", False),
        torch_compile_mode=t.get("torch_compile_mode") or None,
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        warmup_steps=t.get("warmup_steps", 1000),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine_with_min_lr"),
        lr_scheduler_kwargs=t.get("lr_scheduler_kwargs", {"min_lr_rate": 0.1}),
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        fp16=use_fp16,
        bf16=use_bf16,
        # Eval in bf16 too — Trainer's eval defaults to fp32 even when
        # train is mixed-precision, paying 2× on the eval forward pass.
        # Only meaningful when bf16 is on for train.
        bf16_full_eval=use_bf16,
        # Skip extra metric/label compute during eval; we only track
        # eval loss anyway. ~10-20% off each eval call.
        prediction_loss_only=True,
        max_grad_norm=t["max_grad_norm"],
        eval_strategy=eval_strategy,
        eval_steps=t.get("eval_steps", 5000),
        save_strategy=save_strategy,
        save_steps=t.get("save_steps", 5000),
        save_total_limit=t.get("save_total_limit", 3),
        logging_steps=t["logging_steps"],
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        dataloader_num_workers=_resolve_dataloader_workers(t["dataloader_num_workers"]),
        dataloader_pin_memory=t["dataloader_pin_memory"],
        group_by_length=t.get("group_by_length", False),
        optim=optim,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        push_to_hub=hub_model_id is not None,
        hub_model_id=hub_model_id,
        hub_strategy="checkpoint",
    )
