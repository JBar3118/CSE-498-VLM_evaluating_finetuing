import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source = Path("outputs/three_model_comparison/comparison_500.csv")
out = Path("outputs/three_model_comparison/figures")
out.mkdir(parents=True, exist_ok=True)

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))

models = [
    ("Base Qwen", "qwen_base_correct"),
    ("Fine-tuned Qwen", "qwen_finetuned_correct"),
    ("VILA1.5-3B", "vila_correct"),
]

def accuracy(items, field):
    return 100 * sum(r[field] == "True" for r in items) / len(items)

# Figure 1: Overall accuracy
values = [accuracy(rows, field) for _, field in models]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    [name for name, _ in models],
    values
)

ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Overall Model Accuracy on 500 ReVA Questions")

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

fig.tight_layout()
fig.savefig(out / "overall_accuracy.png", dpi=300)
plt.close(fig)

# Figure 2: Category accuracy
categories = [
    "Causal Reasoning",
    "Factual Perception",
    "Spatial Reasoning",
    "Temporal Understanding",
]

grouped = defaultdict(list)
for row in rows:
    grouped[row["category"]].append(row)

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, (name, field) in enumerate(models):
    values = [accuracy(grouped[c], field) for c in categories]
    ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=name
    )

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=12, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Model Accuracy by Reasoning Category")
ax.legend()

fig.tight_layout()
fig.savefig(out / "category_accuracy.png", dpi=300)
plt.close(fig)

print("Saved:", out / "overall_accuracy.png")
print("Saved:", out / "category_accuracy.png")
