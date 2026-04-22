#!/usr/bin/env python3
"""Inference entry point for CLEF 2026 Subtask 1."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from train_subtask1 import (
    configure_logging,
    predict_with_threshold,
    prepare_test_split,
    tokenize_test_frame,
)


LOGGER = logging.getLogger("subtask1_infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for CLEF 2026 Subtask 1.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subtask1_inference"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--use-problem-context", action="store_true")
    return parser.parse_args()
def load_threshold(model_dir: Path) -> float:
    summary_path = model_dir.parent / "metrics" / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if "selected_threshold" in payload:
            return float(payload["selected_threshold"])
    return 0.5


def main() -> None:
    args = parse_args()
    configure_logging()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"

    threshold = float(args.threshold) if args.threshold is not None else load_threshold(args.model_dir)
    LOGGER.info("Using threshold %.4f", threshold)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "tmp"),
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        ),
    )

    test_df = pd.read_json(args.test_path, lines=True)
    prepared_test = prepare_test_split(test_df, args.use_problem_context)
    tokenized_test = tokenize_test_frame(prepared_test, tokenizer, args.max_length)

    prediction_result = predict_with_threshold(trainer, tokenized_test, "test", threshold)
    if "solution_id" not in prepared_test.columns:
        raise ValueError("Test file must contain a 'solution_id' column for CSV submission output.")

    submission = pd.DataFrame(
        {
            "solution_id": prepared_test["solution_id"].astype(str),
            "label": prediction_result.predictions.astype(int),
        }
    )
    csv_path = predictions_dir / "subtask1_predictions.csv"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(csv_path, index=False)
    LOGGER.info("Inference CSV written to %s", csv_path)


if __name__ == "__main__":
    main()
