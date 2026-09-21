"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
from collections import Counter
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))


def get_video_duration(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception as exc:
        print(f"  Warning: could not read duration for {video_path}: {exc}")
        return 0.0


def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
    path = file_path

    prefixes = [
        "#dataset/ReVA_V2/",
        "#dataset/ReVA/",
        "ReVA_V2/",
        "ReVA/",
    ]

    for prefix in prefixes:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    return os.path.join(REVA_ROOT, path)


def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
    path = file_path

    prefixes = [
        "#dataset/ReVA_V2/",
        "#dataset/ReVA/",
        "ReVA_V2/",
        "ReVA/",
    ]

    for prefix in prefixes:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    return os.path.splitext(path)[0]


def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
    if qa.get("qa_id"):
        return str(qa["qa_id"])

    if qa.get("global_index") is not None:
        return f"REVA-G{int(qa['global_index']):06d}"

    video_stem = to_rel_stem(video_item["file_path"])
    safe_stem = video_stem.replace("/", "-")

    return f"REVA-{safe_stem}-{subcategory}-{question_idx:04d}"


def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
    videos = data.get("videos", {})

    for _, video_item in videos.items():
        file_path = video_item.get("file_path", "")
        video_path = to_abs_path(file_path)

        if not os.path.exists(video_path):
            continue

        duration = get_video_duration(video_path)

        if duration <= 0:
            continue

        video_id = to_rel_stem(file_path)
        mcq = video_item.get("mcq", {})

        for _, category_data in mcq.items():
            for question_type, qa_list in category_data.items():
                for question_idx, qa in enumerate(qa_list):
                    options = qa.get("options", {})

                    option_lines = []
                    for label in sorted(options):
                        option_lines.append(f"{label}. {options[label]}")

                    question_text = qa.get("question", "")
                    if option_lines:
                        question_text += "\n" + "\n".join(option_lines)

                    yield {
                        "id": get_qa_id(
                            qa,
                            video_item,
                            question_type,
                            question_idx,
                        ),
                        "video_id": video_id,
                        "duration": duration,
                        "question": question_text,
                        "answer": str(qa.get("correct_answer", "")).strip().upper(),
                        "question_type": question_type,
                    }


def main():
    with open(REVA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    subcats = Counter()
    for item in iter_flat_items(data):
        items.append(item)
        subcats[item.get("question_type", "unknown")] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(items)} QA items to {OUTPUT_JSON}")
    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {subcat}: {count}")


if __name__ == "__main__":
    main()
