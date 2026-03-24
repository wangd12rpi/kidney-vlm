from __future__ import annotations

from typing import Any


def build_training_arguments(train_cfg: Any):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

    report_to = train_cfg.get("report_to", [])
    if report_to is None:
        report_to = []
    elif not isinstance(report_to, list):
        report_to = [str(report_to)]

    return TrainingArguments(
        output_dir=str(train_cfg.output_dir),
        per_device_train_batch_size=int(train_cfg.batch_size),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", train_cfg.batch_size)),
        learning_rate=float(train_cfg.learning_rate),
        num_train_epochs=float(train_cfg.num_epochs),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        remove_unused_columns=False,
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        report_to=report_to,
    )


def build_trainer(
    model: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    data_collator: Any = None,
    args: Any = None,
):
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

    if args is None:
        raise ValueError("Training arguments are required.")

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
