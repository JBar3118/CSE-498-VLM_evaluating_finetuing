import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("outputs")
BASE = ROOT / "qwen_base_500/qwen_base_500/pred.json"
FINETUNED = ROOT / "qwen_finetuned_500/qwen_finetuned_500/pred.json"
VILA = ROOT / "vila_500/outputs.jsonl"

OUT = ROOT / "three_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


from score_reva_predictions import extract_letter


def extract_qwen_answer(text):
    return extract_letter(str(text or ""))


def load_qwen(path):
    records = load_jsonl(path)
    result = {}

    for record in records:
        qid = record["id"]
        if qid in result:
            raise ValueError(f"Duplicate Qwen question ID: {qid}")

        result[qid] = {
            "answer": str(record["answer"]).strip().upper(),
            "prediction": extract_qwen_answer(record.get("pred", "")),
            "raw_response": record.get("pred", ""),
        }

    return result


base = load_qwen(BASE)
finetuned = load_qwen(FINETUNED)

vila = {}
for record in load_jsonl(VILA):
    qid = record["qa_id"]
    if qid in vila:
        raise ValueError(f"Duplicate VILA question ID: {qid}")
    vila[qid] = record

ids = set(base) | set(finetuned) | set(vila)

print("Question counts:")
print("  Base Qwen:", len(base))
print("  Fine-tuned Qwen:", len(finetuned))
print("  VILA:", len(vila))
print("  Union:", len(ids))
print("  Shared:", len(set(base) & set(finetuned) & set(vila)))

if len(ids) != 500:
    raise ValueError("Expected exactly 500 unique question IDs.")

rows = []
patterns = Counter()

for qid in sorted(ids):
    b = base.get(qid, {})
    f = finetuned.get(qid, {})
    v = vila.get(qid, {})

    answers = {
        x for x in [
            b.get("answer"),
            f.get("answer"),
            v.get("correct_answer"),
        ] if x
    }
    if len(answers) != 1:
        raise ValueError(f"Ground-truth mismatch for {qid}: {answers}")

    correct = answers.pop()

    b_pred = b.get("prediction", "")
    f_pred = f.get("prediction", "")
    v_pred = str(v.get("pred_letter") or "").strip().upper()
    if v_pred not in ("A", "B", "C", "D"):
        v_pred = ""

    b_ok = b_pred == correct
    f_ok = f_pred == correct
    v_ok = v_pred == correct

    pattern = (
        ("1" if b_ok else "0")
        + ("1" if f_ok else "0")
        + ("1" if v_ok else "0")
    )
    patterns[pattern] += 1

    rows.append({
        "qa_id": qid,
        "video_id": v.get("video_id", ""),
        "category": v.get("category", ""),
        "subcategory": v.get("subcategory", ""),
        "question": v.get("question", ""),
        "correct_answer": correct,
        "qwen_base_prediction": b_pred,
        "qwen_base_correct": b_ok,
        "qwen_finetuned_prediction": f_pred,
        "qwen_finetuned_correct": f_ok,
        "vila_prediction": v_pred,
        "vila_correct": v_ok,
        "correctness_pattern": pattern,
        "qwen_base_response": b.get("raw_response", ""),
        "qwen_finetuned_response": f.get("raw_response", ""),
        "vila_response": v.get("raw_response", ""),
    })

csv_path = OUT / "comparison_500.csv"
with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nOverall results:")
for label, field in [
    ("Base Qwen", "qwen_base_correct"),
    ("Fine-tuned Qwen", "qwen_finetuned_correct"),
    ("VILA", "vila_correct"),
]:
    count = sum(row[field] for row in rows)
    print(f"  {label}: {count}/500 ({count / 5:.2f}%)")

print("\nCorrectness patterns (Base / Fine-tuned / VILA):")
for pattern in sorted(patterns, reverse=True):
    print(f"  {pattern}: {patterns[pattern]}")

print("\nFine-tuning changes:")
print("  Incorrect -> Correct:", patterns["010"] + patterns["011"])
print("  Correct -> Incorrect:", patterns["100"] + patterns["101"])

print("\nUnanswered / unrecognized predictions:")
for field in [
    "qwen_base_prediction",
    "qwen_finetuned_prediction",
    "vila_prediction",
]:
    print(f"  {field}: {sum(not row[field] for row in rows)}")

print("\nSaved:", csv_path)
