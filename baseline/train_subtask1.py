#!/usr/bin/env python3
"""Training entry point for CLEF 2026 Subtask 1."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LOGGER = logging.getLogger("subtask1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a sequence classifier for CLEF 2026 Subtask 1."
    )
    parser.add_argument("--train-path", type=Path, default=Path("subtask1_dataset/training.jsonl"))
    parser.add_argument("--validation-path", type=Path, default=Path("subtask1_dataset/validation.jsonl"))
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subtask1"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--inner-valid-size", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.2)
    parser.add_argument("--attention-dropout-prob", type=float, default=0.2)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--metric-for-best-model", default="balanced_accuracy")
    parser.add_argument("--threshold-opt-metric", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    parser.add_argument("--threshold-min", type=float, default=0.2)
    parser.add_argument("--threshold-max", type=float, default=0.8)
    parser.add_argument("--threshold-steps", type=int, default=31)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--use-problem-context", action="store_true")
    parser.add_argument("--disable-threshold-calibration", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return path


def normalize_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (0, 1) else None
    if isinstance(value, (float, np.floating)):
        integer_value = int(value)
        return integer_value if integer_value in (0, 1) and float(value) == float(integer_value) else None

    text = str(value).strip().lower()
    if text in {"0", "human", "person", "manual", "false"}:
        return 0
    if text in {"1", "ai", "llm", "model", "machine", "true"}:
        return 1
    return None


def derive_label(row: pd.Series) -> int | None:
    if "label" in row:
        normalized = normalize_label(row.get("label"))
        if normalized is not None:
            return normalized

    for key in ("generator", "detailed_generator"):
        source = str(row.get(key, "")).strip().lower()
        if source == "human":
            return 0
        if source in {"llm", "ai", "assistant", "model"}:
            return 1
        if source.startswith(("gpt", "claude", "gemini", "llama", "mistral", "qwen", "deepseek", "k2")):
            return 1
    return None


def normalize_text(text: Any) -> str:
    text = str(text).replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def build_text(row: pd.Series, use_problem_context: bool) -> str:
    solution = normalize_text(row.get("solution", ""))
    problem = normalize_text(row.get("problem", ""))
    if use_problem_context and problem:
        return f"Problem: {problem}\n\nSolution: {solution}"
    return solution


def metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["id", "problem_id", "solution_id", "generator", "detailed_generator", "language"]
        if column in frame.columns
    ]


def prepare_supervised_split(
    frame: pd.DataFrame,
    split_name: str,
    seed: int,
    use_problem_context: bool,
) -> pd.DataFrame:
    working = frame.copy()
    working["text"] = working.apply(build_text, axis=1, use_problem_context=use_problem_context)
    working["label"] = working.apply(derive_label, axis=1)
    working = working[working["text"].str.len() > 0].copy()
    working = working[working["label"].isin([0, 1])].copy()
    working["label"] = working["label"].astype(int)

    keep_cols = ["text", "label", *metadata_columns(working)]
    working = working[keep_cols].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if working["label"].nunique() < 2:
        raise ValueError(f"{split_name} must contain both classes.")

    LOGGER.info("%s rows: %s", split_name, len(working))
    LOGGER.info("%s label distribution: %s", split_name, working["label"].value_counts(normalize=True).sort_index().to_dict())
    return working


def prepare_test_split(frame: pd.DataFrame, use_problem_context: bool) -> pd.DataFrame:
    working = frame.copy()
    working["text"] = working.apply(build_text, axis=1, use_problem_context=use_problem_context)
    working["label"] = working.apply(derive_label, axis=1)
    working = working[working["text"].str.len() > 0].copy()

    keep = ["text", *metadata_columns(working)]
    labeled = working[working["label"].isin([0, 1])].copy()
    if not labeled.empty:
        labeled["label"] = labeled["label"].astype(int)
        keep = ["text", "label", *metadata_columns(labeled)]
        return labeled[keep].reset_index(drop=True)
    return working[keep].reset_index(drop=True)


def build_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError("Training split is missing one class.")
    weights = len(train_df) / (len(counts) * counts.astype(float))
    return torch.tensor(weights.values, dtype=torch.float)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels", None)
        if labels is None and "label" in model_inputs:
            labels = model_inputs.pop("label")

        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        if labels is not None and logits is not None:
            if self.class_weights is not None:
                loss_function = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


def get_latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in output_dir.glob("checkpoint-*"):
        suffix = checkpoint.name.split("-")[-1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), checkpoint))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def tokenize_frame(frame: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    removable = [column for column in tokenized.column_names if column not in keep]
    tokenized = tokenized.remove_columns(removable)
    tokenized.set_format("torch")
    return tokenized


def tokenize_test_frame(frame: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    keep = {"input_ids", "attention_mask", "token_type_ids"}
    if "label" in tokenized.column_names:
        keep.add("label")
    removable = [column for column in tokenized.column_names if column not in keep]
    tokenized = tokenized.remove_columns(removable)
    tokenized.set_format("torch")
    return tokenized


def safe_evaluate(trainer: Trainer, eval_dataset: Dataset | None = None, metric_key_prefix: str = "eval") -> dict[str, float]:
    kwargs: dict[str, Any] = {"metric_key_prefix": metric_key_prefix}
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset

    try:
        return trainer.evaluate(**kwargs)
    except RuntimeError as error:
        if "on_train_begin must be called before on_evaluate" in str(error):
            for callback in list(trainer.callback_handler.callbacks):
                if callback.__class__.__name__ == "NotebookProgressCallback":
                    trainer.remove_callback(callback.__class__)
            return trainer.evaluate(**kwargs)
        raise


@dataclass
class PredictionResult:
    metrics: dict[str, Any]
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray | None


def predict_with_threshold(
    trainer: Trainer,
    dataset: Dataset,
    prefix: str,
    threshold: float,
) -> PredictionResult:
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    labels = prediction.label_ids
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_positive_rate": float(np.mean(predictions)),
    }
    if labels is not None:
        metrics[f"{prefix}_accuracy"] = float(accuracy_score(labels, predictions))
        metrics[f"{prefix}_balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
        metrics[f"{prefix}_macro_f1"] = float(f1_score(labels, predictions, average="macro"))
        metrics[f"{prefix}_confusion_matrix"] = confusion_matrix(labels, predictions).tolist()
    return PredictionResult(metrics=metrics, probabilities=probabilities, predictions=predictions, labels=labels)


def calibrate_threshold(probabilities: np.ndarray, labels: np.ndarray, metric_name: str, values: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in values:
        predicted = (probabilities >= threshold).astype(int)
        if metric_name == "macro_f1":
            score = f1_score(labels, predicted, average="macro")
        else:
            score = balanced_accuracy_score(labels, predicted)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def group_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> list[dict[str, Any]]:
    if "label" not in frame.columns or "detailed_generator" not in frame.columns:
        return []
    working = frame.copy().reset_index(drop=True)
    if len(working) != len(predictions):
        return []
    working["prediction"] = predictions
    summary = []
    for name, group in working.groupby(working["detailed_generator"].astype(str).str.lower()):
        summary.append(
            {
                "group": name,
                "count": int(len(group)),
                "accuracy": float(accuracy_score(group["label"], group["prediction"])),
                "balanced_accuracy": float(balanced_accuracy_score(group["label"], group["prediction"])),
            }
        )
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    train_path = ensure_path(args.train_path)
    validation_path = ensure_path(args.validation_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    metrics_dir = args.output_dir / "metrics"

    train_raw = pd.read_json(train_path, lines=True)
    valid_raw = pd.read_json(validation_path, lines=True)

    prepared_train = prepare_supervised_split(train_raw, "train", args.seed, args.use_problem_context)
    prepared_valid = prepare_supervised_split(valid_raw, "validation", args.seed, args.use_problem_context)

    stratify = prepared_train["label"] if prepared_train["label"].nunique() > 1 else None
    train_df, inner_valid_df = train_test_split(
        prepared_train,
        test_size=args.inner_valid_size,
        random_state=args.seed,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    inner_valid_df = inner_valid_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_train = tokenize_frame(train_df, tokenizer, args.max_length)
    tokenized_inner_valid = tokenize_frame(inner_valid_df, tokenizer, args.max_length)
    tokenized_valid = tokenize_frame(prepared_valid, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.config.hidden_dropout_prob = args.hidden_dropout_prob
    model.config.attention_probs_dropout_prob = args.attention_dropout_prob
    model.config.classifier_dropout = args.classifier_dropout

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": build_training_arguments(args),
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_inner_valid,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(class_weights=build_class_weights(train_df), **trainer_kwargs)

    checkpoint_dir = args.output_dir / "checkpoints"
    if args.resume_from_checkpoint == "latest":
        resume_checkpoint = get_latest_checkpoint(checkpoint_dir)
    else:
        resume_checkpoint = args.resume_from_checkpoint

    LOGGER.info("Starting training. Resume checkpoint: %s", resume_checkpoint)
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    training_metrics = trainer.state.log_history
    write_json(metrics_dir / "training_log_history.json", training_metrics)

    inner_metrics = safe_evaluate(trainer, metric_key_prefix="inner")
    validation_metrics = safe_evaluate(trainer, eval_dataset=tokenized_valid, metric_key_prefix="validation")

    threshold_grid = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    validation_default = predict_with_threshold(trainer, tokenized_valid, "validation_threshold", 0.5)

    threshold = 0.5
    threshold_score = None
    if not args.disable_threshold_calibration and validation_default.labels is not None:
        threshold, threshold_score = calibrate_threshold(
            validation_default.probabilities,
            validation_default.labels,
            args.threshold_opt_metric,
            threshold_grid,
        )
        LOGGER.info("Calibrated threshold %.4f using %s=%.4f", threshold, args.threshold_opt_metric, threshold_score)

    validation_calibrated = predict_with_threshold(trainer, tokenized_valid, "validation_calibrated", threshold)

    summary = {
        "config": serialize_config(args),
        "paths": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "model_dir": str(model_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "dataset_sizes": {
            "train_after_split": len(train_df),
            "inner_validation": len(inner_valid_df),
            "validation": len(prepared_valid),
        },
        "inner_metrics": inner_metrics,
        "validation_metrics": validation_metrics,
        "validation_threshold_metrics": validation_default.metrics,
        "validation_calibrated_metrics": validation_calibrated.metrics,
        "validation_group_metrics": group_metrics(prepared_valid, validation_calibrated.predictions),
        "selected_threshold": threshold,
        "threshold_score": threshold_score,
    }
    write_json(metrics_dir / "summary.json", summary)
    LOGGER.info("Finished. Model saved to %s", model_dir)


if __name__ == "__main__":
    main()
