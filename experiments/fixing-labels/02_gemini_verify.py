"""
Step 2: Verify audio-only / visual-only labels using Gemini 3 Flash.

For each unique video in samples.csv, reads the local video file from
/tmp/vggsound/video/{video_id}.mp4, uploads it to Gemini Files API, and asks
Gemini to verify label correctness and modality.

Supports parallel processing with a global RPM limiter.
"""

import argparse
import csv
import json
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).parent
MODEL = "gemini-3-flash-preview"
DEFAULT_VIDEO_DIR = Path("/tmp/vggsound/video")


class RateLimiter:
    """Simple thread-safe requests-per-minute limiter."""

    def __init__(self, rpm: int):
        self.rpm = max(0, rpm)
        self._events = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and (now - self._events[0]) >= 60.0:
                    self._events.popleft()
                if len(self._events) < self.rpm:
                    self._events.append(now)
                    return
                sleep_for = max(0.05, 60.0 - (now - self._events[0]))
            time.sleep(sleep_for)


def load_all_classes() -> list[str]:
    classes_csv = (
        Path(__file__).resolve().parent.parent.parent
        / "vggsounder"
        / "data"
        / "classes.csv"
    )
    classes = []
    with open(classes_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row["display_name"])
    return classes


def load_samples(input_path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups[row["video_id"]].append(row)
    return groups


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["video_id"])
    return done


def get_local_video(video_id: str, video_dir: Path) -> str | None:
    path = video_dir / f"{video_id}.mp4"
    if path.exists():
        return str(path)
    return None


def upload_to_gemini(client: genai.Client, video_path: str) -> types.File | None:
    try:
        uploaded = client.files.upload(file=video_path)
        while uploaded.state == "PROCESSING":
            time.sleep(1.0)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state == "ACTIVE":
            return uploaded
        return None
    except Exception:
        return None


def build_prompt(labels_with_modality: list[dict], all_classes: list[str]) -> str:
    label_lines = []
    for entry in labels_with_modality:
        mod_explanation = (
            "audible only (not visible)"
            if entry["modality"] == "A"
            else "visible only (not audible)"
        )
        label_lines.append(f'  - "{entry["label"]}" — labeled as {mod_explanation}')
    labels_block = "\n".join(label_lines)
    classes_block = ", ".join(f'"{c}"' for c in all_classes)
    return f"""Analyze this 10-second video clip carefully. It has labels:

{labels_block}

For EACH label:
1) label_correct: true/false
2) modality_correct: true/false
3) suggested_modality: "A" | "V" | "AV" | ""
4) reason: short explanation

Suggest additional labels ONLY from:
{classes_block}

Return valid JSON exactly:
{{
  "verifications": [
    {{
      "label": "...",
      "label_correct": true,
      "modality_correct": true,
      "suggested_modality": "A",
      "reason": "..."
    }}
  ],
  "additional_labels": [
    {{
      "label": "...",
      "modality": "A"
    }}
  ]
}}"""


def verify_video(
    client: genai.Client,
    gemini_file: types.File,
    labels_with_modality: list[dict],
    all_classes: list[str],
) -> dict | None:
    prompt = build_prompt(labels_with_modality, all_classes)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[gemini_file, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception:
        return None


def process_result(video_id: str, rows: list[dict], result: dict) -> list[dict]:
    output_rows = []
    verifications = result.get("verifications", [])
    additional = result.get("additional_labels", [])

    for v in verifications:
        suggested_labels_json = (
            json.dumps(
                [
                    {"label": a["label"], "modality": a.get("modality", "")}
                    for a in additional
                ]
            )
            if additional
            else "[]"
        )
        original_mod = ""
        for r in rows:
            if r["label"] == v.get("label", ""):
                original_mod = r["modality"]
                break
        output_rows.append(
            {
                "video_id": video_id,
                "label": v.get("label", ""),
                "modality": original_mod,
                "label_correct": v.get("label_correct", True),
                "modality_correct": v.get("modality_correct", True),
                "suggested_labels": suggested_labels_json,
                "suggested_modality": v.get("suggested_modality", ""),
                "reason": v.get("reason", ""),
            }
        )
    return output_rows


def worker(
    video_id: str,
    rows: list[dict],
    video_dir: Path,
    all_classes: list[str],
    api_key: str,
    limiter: RateLimiter,
) -> tuple[str, list[dict], str]:
    video_path = get_local_video(video_id, video_dir)
    if not video_path:
        return video_id, [], "missing_video"

    client = genai.Client(api_key=api_key)
    gemini_file = upload_to_gemini(client, video_path)
    if not gemini_file:
        return video_id, [], "upload_failed"

    labels_with_modality = [
        {"label": r["label"], "modality": r["modality"]} for r in rows
    ]
    limiter.wait()
    result = verify_video(client, gemini_file, labels_with_modality, all_classes)

    try:
        client.files.delete(name=gemini_file.name)
    except Exception:
        pass

    if not result:
        return video_id, [], "generation_failed"
    return video_id, process_result(video_id, rows, result), "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Verify VGGSounder labels with Gemini 3 Flash"
    )
    parser.add_argument("-i", "--input", type=Path, default=SCRIPT_DIR / "samples.csv")
    parser.add_argument(
        "-o", "--output", type=Path, default=SCRIPT_DIR / "gemini_proposals.csv"
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--rpm",
        type=int,
        default=5,
        help="Global requests per minute cap for generate_content (set to your project limit)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")

    all_classes = load_all_classes()
    samples = load_samples(args.input)
    done_ids = load_done_ids(args.output)

    video_ids = [vid for vid in samples if vid not in done_ids]
    if args.limit:
        video_ids = video_ids[: args.limit]

    missing = sum(
        1 for vid in video_ids if not (args.video_dir / f"{vid}.mp4").exists()
    )
    print(f"Model: {MODEL}")
    print(f"Video directory: {args.video_dir}")
    print(f"To process: {len(video_ids)} videos ({missing} missing locally)")
    print(f"Workers: {args.workers}, RPM cap: {args.rpm}")

    fieldnames = [
        "video_id",
        "label",
        "modality",
        "label_correct",
        "modality_correct",
        "suggested_labels",
        "suggested_modality",
        "reason",
    ]
    write_header = not args.output.exists()
    limiter = RateLimiter(args.rpm)

    success = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                worker, vid, samples[vid], args.video_dir, all_classes, api_key, limiter
            ): vid
            for vid in video_ids
        }
        for idx, fut in enumerate(as_completed(futures), start=1):
            video_id, output_rows, status = fut.result()
            if status == "ok" and output_rows:
                with open(args.output, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerows(output_rows)
                success += 1
            else:
                failures += 1
            print(f"[{idx}/{len(video_ids)}] {video_id}: {status}")

    print(f"Done. success={success}, failed={failures}")


if __name__ == "__main__":
    main()
