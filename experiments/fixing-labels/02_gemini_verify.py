"""
Step 2: Verify audio-only / visual-only labels using Gemini 3 Flash.

For each unique video in samples.csv, reads the local video file from
/tmp/vggsound/video/{video_id}.mp4, uploads it to the Gemini Files API,
and asks the model to verify whether each label is correct and whether the
modality assignment is accurate.

Outputs gemini_proposals.csv with the same schema as cleaned_wrong_labels.csv,
so it can be consumed by the review app.

Usage:
    python 02_gemini_verify.py [--input samples.csv] [--output gemini_proposals.csv] [--delay 2]

Environment:
    GEMINI_API_KEY or GOOGLE_API_KEY must be set.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).parent
MODEL = "gemini-3-flash-preview"
VIDEO_DIR = Path("/tmp/vggsound/video")


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
    """Group samples by video_id, returning {video_id: [row_dicts]}."""
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


def get_local_video(video_id: str, video_dir: Path = VIDEO_DIR) -> str | None:
    path = video_dir / f"{video_id}.mp4"
    if path.exists():
        return str(path)
    print(f"  [WARN] Video not found: {path}")
    return None


def upload_to_gemini(client: genai.Client, video_path: str) -> types.File | None:
    try:
        uploaded = client.files.upload(file=video_path)
        while uploaded.state == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state == "ACTIVE":
            return uploaded
        print(f"  [WARN] File state: {uploaded.state}")
        return None
    except Exception as e:
        print(f"  [WARN] Upload failed: {e}")
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

    return f"""Analyze this 10-second video clip carefully. It has been given the following labels:

{labels_block}

For EACH label above, determine:
1. "label_correct" (boolean): Is this label actually present in the video at all?
2. "modality_correct" (boolean): Is the modality assignment correct? (A = only audible not visible, V = only visible not audible)
3. "suggested_modality": What should the correct modality be? One of: "A" (audible only), "V" (visible only), "AV" (both audible and visible), or "" (not present at all).
4. "reason" (string): Brief explanation of your verdict.

Additionally, if you notice sounds or visual events NOT covered by the existing labels, suggest new labels ONLY from this allowed list:
{classes_block}

For each suggested new label provide:
- "label": the label string (must be from the allowed list above)
- "modality": one of "A", "V", or "AV"

Return your answer as valid JSON with this exact structure:
{{
  "verifications": [
    {{
      "label": "...",
      "label_correct": true/false,
      "modality_correct": true/false,
      "suggested_modality": "A" | "V" | "AV" | "",
      "reason": "..."
    }}
  ],
  "additional_labels": [
    {{
      "label": "...",
      "modality": "A" | "V" | "AV"
    }}
  ]
}}

Be conservative: only mark a label as incorrect if you are confident. Only suggest additional labels if clearly present."""


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
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [WARN] Gemini call failed: {e}")
        return None


def process_result(
    video_id: str,
    rows: list[dict],
    result: dict,
) -> list[dict]:
    """Convert Gemini response into CSV rows matching cleaned_wrong_labels.csv schema."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Verify VGGSounder labels with Gemini 3 Flash"
    )
    parser.add_argument("-i", "--input", type=Path, default=SCRIPT_DIR / "samples.csv")
    parser.add_argument(
        "-o", "--output", type=Path, default=SCRIPT_DIR / "gemini_proposals.csv"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help="Directory containing {video_id}.mp4 files (default: /tmp/vggsound/video)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between API calls"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max videos to process (for testing)"
    )
    args = parser.parse_args()
    video_dir = args.video_dir

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    all_classes = load_all_classes()
    samples = load_samples(args.input)
    done_ids = load_done_ids(args.output)

    video_ids = [vid for vid in samples if vid not in done_ids]
    total = len(video_ids)
    if args.limit:
        video_ids = video_ids[: args.limit]

    missing = sum(1 for vid in video_ids if not (video_dir / f"{vid}.mp4").exists())

    print(f"Video directory: {video_dir}")
    print(f"Total unique videos: {len(samples)}")
    print(f"Already processed: {len(done_ids)}")
    print(
        f"Remaining: {total} (processing {len(video_ids)}, {missing} missing locally)"
    )

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

    for i, video_id in enumerate(video_ids):
        rows = samples[video_id]
        labels_with_modality = [
            {"label": r["label"], "modality": r["modality"]} for r in rows
        ]

        print(f"[{i+1}/{len(video_ids)}] {video_id} ({len(rows)} labels)")

        video_path = get_local_video(video_id, video_dir)
        if not video_path:
            continue

        gemini_file = upload_to_gemini(client, video_path)
        if not gemini_file:
            continue

        result = verify_video(client, gemini_file, labels_with_modality, all_classes)
        if not result:
            continue

        output_rows = process_result(video_id, rows, result)

        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(output_rows)

        try:
            client.files.delete(name=gemini_file.name)
        except Exception:
            pass

        if i < len(video_ids) - 1:
            time.sleep(args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
