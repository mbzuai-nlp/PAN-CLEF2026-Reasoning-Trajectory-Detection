#!/usr/bin/env python3
"""
Format Checker for Subtask 1 and Subtask 2 submissions.

Usage:
    python format_checker.py --subtask 1 --file submission.csv
    python format_checker.py --subtask 2 --file submission.csv

Extra use (add on at the end):
    --ref_file path/to/test_labels.jsonl   (to check IDs against ground truth)
"""

import os
import sys
import csv
import json
import argparse


GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):   print(f"  {RED}✘{RESET}  {msg}")
def info(msg):  print(f"     {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}")



VALID_LABELS     = {"safe", "unsafe", "potentially unsafe"}
BINARY_LABELS    = {"safe", "unsafe"}

def coerce_label(raw):
    if not isinstance(raw, str):
        return None
    return raw.strip().lower()


def parse_detailed_label(raw):
    """
    Try to parse detailed_label from pipe-separated or JSON array string.
    Returns (values, error_message).
    values is None on failure.
    """
    if not isinstance(raw, str):
        return None, "Not a string"

    raw = raw.strip()

    # JSON array
    if raw.startswith("[") and raw.endswith("]"):
        try:
            items = json.loads(raw)
            values = [float(x) for x in items]
            for v in values:
                if not (0.0 <= v <= 1.0):
                    return None, f"Value out of range [0,1]: {v}"
            return values, None
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return None, f"JSON parse error: {e}"

    # Pipe-separated
    parts = [x.strip() for x in raw.split("|") if x.strip() != ""]
    if not parts:
        return None, "Empty detailed_label"
    try:
        values = [float(x) for x in parts]
        for v in values:
            if not (0.0 <= v <= 1.0):
                return None, f"Value out of range [0,1]: {v}"
        return values, None
    except ValueError as e:
        return None, f"Parse error: {e}"


def load_ref_ids(ref_file):
    """Load expected IDs from a reference JSONL file."""
    ids = set()
    if not ref_file or not os.path.isfile(ref_file):
        return None
    with open(ref_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                rid = item.get("id")
                if rid:
                    ids.add(str(rid).strip())
            except json.JSONDecodeError:
                continue
    return ids


def load_csv(file_path):
    """Load CSV and return (fieldnames, rows). Raises on file error."""
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return [f.strip() for f in fieldnames], rows


# ---------------------------------------------------------------------------
# Subtask 1 checker
# ---------------------------------------------------------------------------

# Expected columns for subtask 1:
#   ID, label
# label: "human" or "ai" (case insensitive)

S1_REQUIRED_COLS  = ["ID", "label"]
S1_VALID_LABELS   = {"human", "ai"}

def check_subtask1(file_path, ref_ids):
    issues = 0
    warnings = 0

    # --- File loading ---
    header("[ Step 1 ] Loading file")
    try:
        fieldnames, rows = load_csv(file_path)
        ok(f"File loaded: {len(rows)} rows found")
    except FileNotFoundError:
        err(f"File not found: {file_path}")
        return
    except Exception as e:
        err(f"Failed to read CSV: {e}")
        return

    # --- Column check ---
    header("[ Step 2 ] Checking columns")
    missing_cols = [c for c in S1_REQUIRED_COLS if c not in fieldnames]
    extra_cols   = [c for c in fieldnames if c not in S1_REQUIRED_COLS]

    if not missing_cols:
        ok(f"All required columns present: {S1_REQUIRED_COLS}")
    else:
        for c in missing_cols:
            err(f"Missing required column: '{c}'")
        issues += len(missing_cols)

    if extra_cols:
        for c in extra_cols:
            warn(f"Unexpected extra column (will be ignored): '{c}'")
        warnings += len(extra_cols)

    if missing_cols:
        err("Cannot continue checks — required columns are missing.")
        _summary(issues, warnings)
        return

    # --- Row-level checks ---
    header("[ Step 3 ] Checking rows")

    seen_ids        = {}
    missing_id_rows = []
    duplicate_ids   = []
    bad_label_rows  = []
    label_counts    = {"human": 0, "ai": 0}

    for i, row in enumerate(rows, start=2):
        rid   = row.get("ID", "").strip()
        label = row.get("label", "")

        # --- ID ---
        if not rid:
            missing_id_rows.append(i)
            issues += 1
            continue

        if rid in seen_ids:
            duplicate_ids.append((rid, seen_ids[rid], i))
            issues += 1
        else:
            seen_ids[rid] = i

        # --- label ---
        norm = label.strip().lower() if isinstance(label, str) else ""
        if norm not in S1_VALID_LABELS:
            bad_label_rows.append((rid, label))
            issues += 1
        else:
            label_counts[norm] += 1

    if not missing_id_rows:
        ok("No rows with missing ID")
    else:
        err(f"{len(missing_id_rows)} row(s) with missing ID (CSV rows: {missing_id_rows[:10]}{'...' if len(missing_id_rows)>10 else ''})")

    if not duplicate_ids:
        ok("No duplicate IDs")
    else:
        err(f"{len(duplicate_ids)} duplicate ID(s):")
        for rid, first, second in duplicate_ids[:5]:
            info(f"  ID '{rid}' at rows {first} and {second}")
        if len(duplicate_ids) > 5:
            info(f"  ... and {len(duplicate_ids)-5} more")

    if not bad_label_rows:
        ok(f"All labels are valid {sorted(S1_VALID_LABELS)}")
    else:
        err(f"{len(bad_label_rows)} row(s) with invalid label:")
        for rid, lbl in bad_label_rows[:5]:
            info(f"  ID '{rid}': got '{lbl}' — expected one of {sorted(S1_VALID_LABELS)} (case insensitive)")
        if len(bad_label_rows) > 5:
            info(f"  ... and {len(bad_label_rows)-5} more")

    # Label distribution
    total_valid = sum(label_counts.values())
    if total_valid > 0:
        ok(f"Label distribution — human: {label_counts['human']} ({100*label_counts['human']/total_valid:.1f}%),"
           f"  ai: {label_counts['ai']} ({100*label_counts['ai']/total_valid:.1f}%)")
        if label_counts["human"] == 0:
            warn("No 'human' predictions — F1(human) will be 0")
            warnings += 1
        if label_counts["ai"] == 0:
            warn("No 'ai' predictions — F1(ai) will be 0")
            warnings += 1

    # --- ID coverage check ---
    if ref_ids is not None:
        header("[ Step 4 ] Checking ID coverage against reference")
        submitted_ids    = set(seen_ids.keys())
        missing_from_sub = ref_ids - submitted_ids
        extra_in_sub     = submitted_ids - ref_ids

        if not missing_from_sub:
            ok(f"All {len(ref_ids)} reference IDs are present in submission")
        else:
            err(f"{len(missing_from_sub)} ID(s) in reference but missing from submission:")
            for rid in sorted(missing_from_sub)[:10]:
                info(f"  {rid}")
            if len(missing_from_sub) > 10:
                info(f"  ... and {len(missing_from_sub)-10} more")
            issues += len(missing_from_sub)

        if extra_in_sub:
            warn(f"{len(extra_in_sub)} ID(s) in submission but not in reference (will be ignored):")
            for rid in sorted(extra_in_sub)[:5]:
                info(f"  {rid}")
            if len(extra_in_sub) > 5:
                info(f"  ... and {len(extra_in_sub)-5} more")
            warnings += len(extra_in_sub)
        else:
            ok("No extra IDs beyond the reference set")
    else:
        header("[ Step 4 ] ID coverage check")
        warn("No --ref_file provided — skipping ID coverage check")
        warnings += 1

    _summary(issues, warnings)


# ---------------------------------------------------------------------------
# Subtask 2 checker
# ---------------------------------------------------------------------------

S2_REQUIRED_COLS = ["ID", "label", "detailed_label"]

def check_subtask2(file_path, ref_ids):
    issues = 0
    warnings = 0

    # --- File loading ---
    header("[ Step 1 ] Loading file")
    try:
        fieldnames, rows = load_csv(file_path)
        ok(f"File loaded: {len(rows)} rows found")
    except FileNotFoundError:
        err(f"File not found: {file_path}")
        return
    except Exception as e:
        err(f"Failed to read CSV: {e}")
        return

    # --- Column check ---
    header("[ Step 2 ] Checking columns")
    missing_cols = [c for c in S2_REQUIRED_COLS if c not in fieldnames]
    extra_cols   = [c for c in fieldnames if c not in S2_REQUIRED_COLS]

    if not missing_cols:
        ok(f"All required columns present: {S2_REQUIRED_COLS}")
    else:
        for c in missing_cols:
            err(f"Missing required column: '{c}'")
        issues += len(missing_cols)

    if extra_cols:
        for c in extra_cols:
            warn(f"Unexpected extra column (will be ignored): '{c}'")
        warnings += len(extra_cols)

    if missing_cols:
        err("Cannot continue checks — required columns are missing.")
        _summary(issues, warnings)
        return

    # --- Row-level checks ---
    header("[ Step 3 ] Checking rows")

    seen_ids             = {}
    missing_id_rows      = []
    duplicate_ids        = []
    bad_label_rows       = []
    coerced_label_rows   = []
    bad_detailed_rows    = []
    empty_detailed_rows  = []

    for i, row in enumerate(rows, start=2):
        rid    = row.get("ID", "").strip()
        label  = row.get("label", "")
        detail = row.get("detailed_label", "")

        # --- ID ---
        if not rid:
            missing_id_rows.append(i)
            issues += 1
            continue

        if rid in seen_ids:
            duplicate_ids.append((rid, seen_ids[rid], i))
            issues += 1
        else:
            seen_ids[rid] = i

        # --- label ---
        norm = coerce_label(label)
        if norm not in VALID_LABELS:
            bad_label_rows.append((rid, label))
            issues += 1
        elif norm == "potentially unsafe":
            coerced_label_rows.append(rid)   # valid but will be mapped

        # --- detailed_label ---
        if detail is None or str(detail).strip() == "":
            empty_detailed_rows.append(rid)
            warnings += 1
        else:
            values, parse_err = parse_detailed_label(str(detail))
            if parse_err:
                bad_detailed_rows.append((rid, str(detail)[:40], parse_err))
                issues += 1

    # ID issues
    if not missing_id_rows:
        ok("No rows with missing ID")
    else:
        err(f"{len(missing_id_rows)} row(s) with missing ID (CSV rows: {missing_id_rows[:10]}{'...' if len(missing_id_rows)>10 else ''})")

    if not duplicate_ids:
        ok("No duplicate IDs")
    else:
        err(f"{len(duplicate_ids)} duplicate ID(s):")
        for rid, first, second in duplicate_ids[:5]:
            info(f"  ID '{rid}' at rows {first} and {second}")
        if len(duplicate_ids) > 5:
            info(f"  ... and {len(duplicate_ids)-5} more")

    # label issues
    if not bad_label_rows:
        ok(f"All labels are valid {sorted(VALID_LABELS)}")
    else:
        err(f"{len(bad_label_rows)} row(s) with invalid label:")
        for rid, lbl in bad_label_rows[:5]:
            info(f"  ID '{rid}': got '{lbl}' — expected one of {sorted(VALID_LABELS)}")
        if len(bad_label_rows) > 5:
            info(f"  ... and {len(bad_label_rows)-5} more")

    if coerced_label_rows:
        warn(f"{len(coerced_label_rows)} row(s) with 'potentially unsafe' label — will be mapped to 'unsafe' during scoring:")
        for rid in coerced_label_rows[:5]:
            info(f"  ID '{rid}'")
        if len(coerced_label_rows) > 5:
            info(f"  ... and {len(coerced_label_rows)-5} more")
        warnings += len(coerced_label_rows)

    # detailed_label issues
    if not empty_detailed_rows:
        ok("All rows have a non-empty detailed_label")
    else:
        warn(f"{len(empty_detailed_rows)} row(s) with empty detailed_label (will score 0 for s2_soft_f1):")
        for rid in empty_detailed_rows[:5]:
            info(f"  ID '{rid}'")
        if len(empty_detailed_rows) > 5:
            info(f"  ... and {len(empty_detailed_rows)-5} more")

    if not bad_detailed_rows:
        ok("All detailed_labels parsed successfully")
    else:
        err(f"{len(bad_detailed_rows)} row(s) with unparseable detailed_label:")
        for rid, raw, msg in bad_detailed_rows[:5]:
            info(f"  ID '{rid}': '{raw}' → {msg}")
        if len(bad_detailed_rows) > 5:
            info(f"  ... and {len(bad_detailed_rows)-5} more")

    # --- ID coverage check ---
    if ref_ids is not None:
        header("[ Step 4 ] Checking ID coverage against reference")
        submitted_ids    = set(seen_ids.keys())
        missing_from_sub = ref_ids - submitted_ids
        extra_in_sub     = submitted_ids - ref_ids

        if not missing_from_sub:
            ok(f"All {len(ref_ids)} reference IDs are present in submission")
        else:
            err(f"{len(missing_from_sub)} ID(s) in reference but missing from submission:")
            for rid in sorted(missing_from_sub)[:10]:
                info(f"  {rid}")
            if len(missing_from_sub) > 10:
                info(f"  ... and {len(missing_from_sub)-10} more")
            issues += len(missing_from_sub)

        if extra_in_sub:
            warn(f"{len(extra_in_sub)} ID(s) in submission but not in reference (will be ignored by scorer):")
            for rid in sorted(extra_in_sub)[:5]:
                info(f"  {rid}")
            if len(extra_in_sub) > 5:
                info(f"  ... and {len(extra_in_sub)-5} more")
            warnings += len(extra_in_sub)
        else:
            ok("No extra IDs beyond the reference set")
    else:
        header("[ Step 4 ] ID coverage check")
        warn("No --ref_file provided — skipping ID coverage check")
        warnings += 1

    _summary(issues, warnings)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(issues, warnings):
    header("[ Summary ]")
    if issues == 0 and warnings == 0:
        print(f"  {GREEN}{BOLD}✔ Submission looks good! No issues found.{RESET}")
    elif issues == 0:
        print(f"  {YELLOW}{BOLD}⚠ {warnings} warning(s) — submission is valid but review warnings above.{RESET}")
    else:
        print(f"  {RED}{BOLD}✘ {issues} error(s), {warnings} warning(s) — please fix errors before submitting.{RESET}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Format checker for Subtask 1 and Subtask 2 submissions."
    )
    parser.add_argument(
        "--subtask", required=True, choices=["1", "2"],
        help="Which subtask to check (1 or 2)"
    )
    parser.add_argument(
        "--file", required=True,
        help="Path to your submission CSV file"
    )
    parser.add_argument(
        "--ref_file", default=None,
        help="(Optional) Path to reference JSONL file to validate IDs"
    )
    args = parser.parse_args()

    print(f"\n{BOLD}=== Format Checker — Subtask {args.subtask} ==={RESET}")
    print(f"  File : {args.file}")
    if args.ref_file:
        print(f"  Ref  : {args.ref_file}")

    if os.path.basename(args.file) != "submission.csv":
        err(f"Invalid file name '{os.path.basename(args.file)}' — must be exactly 'submission.csv'")
        _summary(1, 0)
        sys.exit(1)

    ref_ids = load_ref_ids(args.ref_file)
    if ref_ids is not None:
        ok(f"Loaded {len(ref_ids)} reference IDs from {args.ref_file}")

    if args.subtask == "1":
        check_subtask1(args.file, ref_ids)
    else:
        check_subtask2(args.file, ref_ids)


if __name__ == "__main__":
    main()