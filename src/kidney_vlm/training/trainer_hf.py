from __future__ import annotations

from typing import Any


def build_training_arguments(train_cfg: Any):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install project dependencies first.") from exc

    return TrainingArguments(
        output_dir=str(train_cfg.output_dir),
        per_device_train_batch_size=int(train_cfg.batch_size),
        learning_rate=float(train_cfg.learning_rate),
        num_train_epochs=float(train_cfg.num_epochs),
        remove_unused_columns=False,
        logging_steps=10,
        report_to=[],
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
