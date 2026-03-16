"""
Step 1: Extract all audio-only and visual-only labeled entries from VGGSounder.

Produces samples.csv with the same schema as vggsounder+background-music.csv,
containing only rows where the label modality is "A" (audio-only) or "V" (visual-only).
"""

import argparse
import csv
from pathlib import Path

from vggsounder import VGGSounder


def extract_samples(output_path: Path) -> None:
    rows = []

    for modality_filter in ("A ONLY", "V ONLY"):
        vgg = VGGSounder(modality=modality_filter, background_music=None)
        for video_data in vgg:
            meta = video_data.meta_labels
            for label, mod in zip(video_data.labels, video_data.modalities):
                rows.append(
                    {
                        "video_id": video_data.video_id,
                        "label": label,
                        "modality": mod,
                        "background_music": meta["background_music"],
                        "static_image": meta["static_image"],
                        "voice_over": meta["voice_over"],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "label",
                "modality",
                "background_music",
                "static_image",
                "voice_over",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract A-only and V-only samples from VGGSounder"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "samples.csv",
        help="Output CSV path (default: samples.csv in script directory)",
    )
    args = parser.parse_args()
    extract_samples(args.output)


if __name__ == "__main__":
    main()
