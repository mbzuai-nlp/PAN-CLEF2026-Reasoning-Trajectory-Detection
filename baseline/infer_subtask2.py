#!/usr/bin/env python3
"""Inference entry point for CLEF 2026 Subtask 2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from train_subtask2 import (
    MultiTaskClassifier,
    TRACE_ID2LABEL,
    build_step_samples,
    build_trace_samples,
    configure_logging,
    create_loader,
    evaluate,
    load_jsonl_files,
)
from transformers import AutoTokenizer


LOGGER = logging.getLogger("subtask2_infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for CLEF 2026 Subtask 2.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--test-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subtask2_inference"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskClassifier(args.model_name, args.dropout).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for test_file in args.test_files:
        test_records = load_jsonl_files([test_file], f"test:{test_file.name}")
        if not test_records:
            continue
        if any("solution_id" not in record for record in test_records):
            raise ValueError(f"Test file must contain 'solution_id' for every record: {test_file}")

        step_samples = build_step_samples(test_records)
        trace_samples = build_trace_samples(test_records)
        if not step_samples or not trace_samples:
            LOGGER.warning("Skipping %s because it does not contain usable trace data.", test_file)
            continue

        step_loader = create_loader(
            step_samples,
            tokenizer,
            args.max_length,
            args.batch_size,
            args.num_workers,
            shuffle=False,
        )
        step_eval = evaluate(model, step_loader, device, "step")

        trace_loader = create_loader(
            trace_samples,
            tokenizer,
            args.max_length,
            args.batch_size,
            args.num_workers,
            shuffle=False,
        )
        trace_eval = evaluate(model, trace_loader, device, "trace")

        step_predictions_by_trace: dict[str, list[int]] = {}
        if step_samples:
            for sample, prediction in sorted(
                zip(step_samples, step_eval.predictions),
                key=lambda item: (str(item[0]["trace_id"]), int(item[0]["step_idx"])),
            ):
                trace_id = str(sample["trace_id"])
                step_predictions_by_trace.setdefault(trace_id, []).append(int(prediction))

        solution_id_by_trace = {
            str(record.get("id", "")): str(record["solution_id"])
            for record in test_records
        }

        rows = []
        for sample, prediction in zip(trace_samples, trace_eval.predictions):
            trace_id = str(sample["trace_id"])
            rows.append(
                {
                    "solution_id": solution_id_by_trace.get(trace_id, trace_id),
                    "stepwise_label": json.dumps(step_predictions_by_trace.get(trace_id, []), ensure_ascii=False),
                    "final_label": TRACE_ID2LABEL[prediction],
                }
            )

        predictions_dir.mkdir(parents=True, exist_ok=True)
        csv_path = predictions_dir / f"{test_file.stem}_predictions.csv"
        pd.DataFrame(rows, columns=["solution_id", "stepwise_label", "final_label"]).to_csv(csv_path, index=False)
        LOGGER.info("Inference CSV written to %s", csv_path)


if __name__ == "__main__":
    main()
