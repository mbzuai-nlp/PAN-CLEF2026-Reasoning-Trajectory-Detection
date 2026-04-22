#!/usr/bin/env python3
"""Training entry point for CLEF 2026 Subtask 2."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LOGGER = logging.getLogger("subtask2")

TRACE_LABEL2ID = {"safe": 0, "potentially_unsafe": 1, "unsafe": 2}
TRACE_ID2LABEL = {value: key for key, value in TRACE_LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multi-task classifier for CLEF 2026 Subtask 2."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/subtask2/train"))
    parser.add_argument("--validation-dir", type=Path, default=Path("data/subtask2/validation"))
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subtask2"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--step-balance-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="best_mtl.pt")
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, list):
            serialized[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serialized[key] = value
    return serialized


def parse_reasoning_trace(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    pieces = [part.strip() for part in re.split(r"Step\s+\d+\s*:", text) if part.strip()]
    return pieces or [text]


def infer_trace_label(record: dict[str, Any], source_path: Path) -> str:
    if "label" in record and str(record["label"]).strip():
        label = str(record["label"]).strip().lower()
        if label in TRACE_LABEL2ID:
            return label
    filename = source_path.stem.lower()
    if "potentially" in filename:
        return "potentially_unsafe"
    if "unsafe" in filename:
        return "unsafe"
    return "safe"


def load_jsonl_files(paths: list[Path], split_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            LOGGER.warning("Skipping missing file: %s", path)
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_path"] = str(path)
                record["label"] = infer_trace_label(record, path)
                records.append(record)
    LOGGER.info("%s records loaded for %s", len(records), split_name)
    return records


def discover_jsonl_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(directory.glob("*.jsonl"))


def build_step_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        trace_id = record.get("id", "")
        steps = parse_reasoning_trace(record.get("reasoning_trace", ""))
        detailed_labels = record.get("detailed_label", [])
        for index, step_text in enumerate(steps):
            label = int(detailed_labels[index]) if index < len(detailed_labels) else 0
            samples.append(
                {
                    "text": f"[QUERY] {record.get('query', '')} [STEP] {step_text}",
                    "label": label,
                    "trace_id": trace_id,
                    "step_idx": index,
                }
            )
    return samples


def build_trace_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        reasoning_trace = record.get("reasoning_trace", "")
        if isinstance(reasoning_trace, list):
            trace_text = " ".join(str(item) for item in reasoning_trace)
        else:
            trace_text = str(reasoning_trace)
        samples.append(
            {
                "text": f"[QUERY] {record.get('query', '')} [TRACE] {trace_text}",
                "label": TRACE_LABEL2ID[record["label"]],
                "trace_id": record.get("id", ""),
            }
        )
    return samples


class ClassificationDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.samples[index]
        encoded = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.step_head = nn.Linear(self.encoder.config.hidden_size, 2)
        self.trace_head = nn.Linear(self.encoder.config.hidden_size, 3)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.step_head(pooled) if task == "step" else self.trace_head(pooled)
        loss = self.loss_function(logits, labels.view(-1)) if labels is not None else None
        return loss, logits


@dataclass
class EvalResult:
    task: str
    macro_f1: float
    accuracy: float
    labels: list[int]
    predictions: list[int]


def create_loader(
    samples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ClassificationDataset(samples, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def evaluate(model: MultiTaskClassifier, loader: DataLoader, device: torch.device, task: str) -> EvalResult:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{task}", leave=False):
            _, logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                task=task,
            )
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return EvalResult(task=task, macro_f1=float(macro_f1), accuracy=float(accuracy), labels=labels, predictions=predictions)


def serialize_eval_result(result: EvalResult, label_names: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": result.task,
        "macro_f1": result.macro_f1,
        "accuracy": result.accuracy,
        "confusion_matrix": confusion_matrix(result.labels, result.predictions).tolist(),
    }
    if label_names is not None:
        label_ids = list(range(len(label_names)))
        payload["classification_report"] = classification_report(
            result.labels,
            result.predictions,
            labels=label_ids,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
    return payload


def balance_step_samples(samples: list[dict[str, Any]], trace_sample_count: int, ratio: float) -> list[dict[str, Any]]:
    max_steps = int(trace_sample_count * ratio)
    if max_steps <= 0 or len(samples) <= max_steps:
        return samples
    shuffled = samples[:]
    random.shuffle(shuffled)
    return shuffled[:max_steps]


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl_files(discover_jsonl_files(args.train_dir), "train")
    validation_records = load_jsonl_files(discover_jsonl_files(args.validation_dir), "validation")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskClassifier(args.model_name, args.dropout).to(device)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    train_step_samples = build_step_samples(train_records)
    train_trace_samples = build_trace_samples(train_records)
    validation_step_samples = build_step_samples(validation_records)
    validation_trace_samples = build_trace_samples(validation_records)

    train_step_samples = balance_step_samples(
        train_step_samples,
        trace_sample_count=len(train_trace_samples),
        ratio=args.step_balance_ratio,
    )

    train_step_loader = create_loader(
        train_step_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    train_trace_loader = create_loader(
        train_trace_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    validation_step_loader = create_loader(
        validation_step_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    validation_trace_loader = create_loader(
        validation_trace_samples,
        tokenizer,
        args.max_length,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_step_loader) + len(train_trace_loader)) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    history: list[dict[str, Any]] = []
    best_checkpoint = checkpoints_dir / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = [("step", batch) for batch in train_step_loader] + [("trace", batch) for batch in train_trace_loader]
        random.shuffle(batches)

        for task, batch in tqdm(batches, desc=f"train-epoch-{epoch}"):
            optimizer.zero_grad()
            loss, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                task=task,
                labels=batch["labels"].to(device),
            )
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        step_eval = evaluate(model, validation_step_loader, device, "step")
        trace_eval = evaluate(model, validation_trace_loader, device, "trace")
        mean_f1 = (step_eval.macro_f1 + trace_eval.macro_f1) / 2.0

        epoch_summary = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, len(batches)),
            "step_macro_f1": step_eval.macro_f1,
            "trace_macro_f1": trace_eval.macro_f1,
            "average_macro_f1": mean_f1,
            "step_accuracy": step_eval.accuracy,
            "trace_accuracy": trace_eval.accuracy,
        }
        history.append(epoch_summary)
        LOGGER.info("Epoch summary: %s", epoch_summary)

        epoch_checkpoint = checkpoints_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": serialize_config(args),
                "history": history,
            },
            epoch_checkpoint,
        )

        if mean_f1 > best_score:
            best_score = mean_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": serialize_config(args),
                    "history": history,
                    "best_score": best_score,
                },
                best_checkpoint,
            )
            LOGGER.info("Saved improved checkpoint to %s", best_checkpoint)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    validation_step_eval = evaluate(model, validation_step_loader, device, "step")
    validation_trace_eval = evaluate(model, validation_trace_loader, device, "trace")

    summary = {
        "config": serialize_config(args),
        "best_checkpoint": str(best_checkpoint),
        "best_score": best_score,
        "history": history,
        "validation": {
            "step": serialize_eval_result(validation_step_eval, ["safe_step", "unsafe_step"]),
            "trace": serialize_eval_result(validation_trace_eval, ["safe", "potentially_unsafe", "unsafe"]),
        },
    }

    write_json(metrics_dir / "summary.json", summary)
    write_json(metrics_dir / "training_history.json", history)
    LOGGER.info("Finished. Artifacts written to %s", args.output_dir)


if __name__ == "__main__":
    main()
