#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return text.strip()


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    answer_text = extract_answer(text).strip().upper()

    # If the extracted answer is just a letter
    if re.fullmatch(r"[A-H]", answer_text):
        return answer_text

    # Look for phrases such as "Answer: C" or "Final answer: C"
    match = re.search(r"(?:FINAL\s+ANSWER|ANSWER)\s*:\s*([A-H])\b", answer_text)
    if match:
        return match.group(1)

    # Look for phrases such as "The answer is C"
    match = re.search(r"\bANSWER\s+IS\s+([A-H])\b", answer_text)
    if match:
        return match.group(1)

    return ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    predictions = {}

    for file_path in sorted(output_dir.glob("*.json")):
        if file_path.name == "result.json":
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                if "id" in item:
                    predictions[str(item["id"])] = item

    return predictions


def score_predictions(
    preds: dict[str, dict[str, Any]],
    gt_list: list[dict[str, Any]],
) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    results = {}
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    completed = 0
    total_correct = 0
    total_questions = len(gt_list)

    for gt in gt_list:
        qa_id = str(gt["id"])
        correct_letter = str(gt.get("answer", "")).strip().upper()
        question_type = gt.get("question_type", "unknown")

        pred_item = preds.get(qa_id)

        if pred_item is not None:
            completed += 1
            pred_letter = extract_letter(str(pred_item.get("pred", "")))
        else:
            pred_letter = ""

        acc = int(pred_letter == correct_letter)

        if acc:
            total_correct += 1

        category_stats[question_type]["correct"] += acc
        category_stats[question_type]["total"] += 1

        results[qa_id] = {
            "pred_letter": pred_letter,
            "answer": correct_letter,
            "acc": acc,
            "question_type": question_type,
        }

    completed_pct = (
        100.0 * completed / total_questions
        if total_questions
        else 0.0
    )

    total_pct = (
        100.0 * total_correct / total_questions
        if total_questions
        else 0.0
    )

    lines = [
        f"Completed: {completed}/{total_questions} = {completed_pct:.2f}%",
        f"Total: {total_correct}/{total_questions} = {total_pct:.2f}%",
    ]

    for category in sorted(category_stats):
        correct = category_stats[category]["correct"]
        total = category_stats[category]["total"]
        percentage = 100.0 * correct / total if total else 0.0

        lines.append(
            f"{category}: {correct}/{total} = {percentage:.2f}%"
        )

    csv_text = "\n".join(lines) + "\n"

    return results, csv_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
