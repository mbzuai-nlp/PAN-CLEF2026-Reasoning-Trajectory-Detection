# Baseline Training Scripts

This repository contains standalone Python scripts for training and inference for both CLEF 2026 subtasks.

## Files

- `train_subtask1.py`: Train and validate the Subtask 1 binary classifier.
- `infer_subtask1.py`: Run inference for Subtask 1 using a saved model.
- `train_subtask2.py`: Train and validate the Subtask 2 multi-task model.
- `infer_subtask2.py`: Run inference for Subtask 2 using a saved checkpoint.

## Environment

Create a virtual environment and install the dependencies before running the scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Subtask 1

### Training

```bash
python3 train_subtask1.py \
  --train-path subtask1_dataset/training.jsonl \
  --validation-path subtask1_dataset/validation.jsonl \
  --output-dir outputs/subtask1
```

Training outputs:

- `outputs/subtask1/model`: best saved model and tokenizer
- `outputs/subtask1/checkpoints`: intermediate Trainer checkpoints
- `outputs/subtask1/metrics/summary.json`: validation summary and calibrated threshold
- `outputs/subtask1/metrics/training_log_history.json`: raw training log history

### Inference

```bash
python3 infer_subtask1.py \
  --model-dir outputs/subtask1/model \
  --test-path subtask1_dataset/test.jsonl \
  --output-dir outputs/subtask1_inference
```

Inference outputs: `outputs/subtask1_inference/predictions/subtask1_predictions.csv`

> Notes: If `--threshold` is not provided, the script reads `selected_threshold` from `outputs/subtask1/metrics/summary.json`.

## Subtask 2

### Training

```bash
python3 train_subtask2.py \
  --train-dir data/subtask2/train \
  --validation-dir data/subtask2/validation \
  --output-dir outputs/subtask2
```

Training outputs:

- `outputs/subtask2/checkpoints/best_mtl.pt`: best checkpoint
- `outputs/subtask2/checkpoints/epoch_*.pt`: per-epoch checkpoints
- `outputs/subtask2/tokenizer`: saved tokenizer
- `outputs/subtask2/metrics/summary.json`: validation summary
- `outputs/subtask2/metrics/training_history.json`: per-epoch history

### Inference

```bash
python3 infer_subtask2.py \
  --checkpoint-path outputs/subtask2/checkpoints/best_mtl.pt \
  --test-files data/subtask2/test_a.jsonl data/subtask2/test_b.jsonl \
  --output-dir outputs/subtask2_inference
```

Inference outputs: `outputs/subtask2_inference/predictions/*_predictions.csv`

> Notes: `infer_subtask2.py` reconstructs the model architecture and loads weights from `best_mtl.pt`.

## Recommended Workflow

1. Train the model with `train_subtask1.py` or `train_subtask2.py`.
2. Check the saved validation metrics under the corresponding `metrics` directory.
3. Run `infer_subtask1.py` or `infer_subtask2.py` on test or unseen files.
4. Use the generated CSV files for submission.
