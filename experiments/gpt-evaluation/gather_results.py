import pandas as pd
import argparse
import os
from collections import defaultdict
import ast

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    results = defaultdict(dict)
    for modality in ["a", "v", "av"]:
        df_path = os.path.join("./csv", modality, f"predictions-{args.model}.csv")
        df = pd.read_csv(df_path)

        for _, row in df.iterrows():
            video_id = row["video_id"].split(".")[0]

            try:
                predictions = ast.literal_eval(row["suggestions"])
            except Exception as e:
                raise ValueError(
                    f"Invalid suggestions: {row['suggestions']}, modality: {modality}, video_id: {video_id}. Error: {e}"
                )

            if type(predictions) != dict:
                continue

            if args.threshold is not None:
                predictions = [
                    k
                    for k, pred in predictions.items()
                    if pred["score"] >= args.threshold
                ]
            else:
                predictions = [
                    k for k, pred in predictions.items() if pred["pred"] == "yes"
                ]

            results[video_id][modality] = predictions + [""] * (10 - len(predictions))

    # drop videos that have no predictions for at least one modality
    results = {
        video_id: results[video_id]
        for video_id in results
        if all(modality in results[video_id] for modality in ["a", "v", "av"])
    }

    import pickle

    # Build the output dict in the required format
    out_dict = {}
    for video_id in results.keys():
        out_dict[video_id] = {
            "predictions": {
                "a": results[video_id].get("a", []),
                "v": results[video_id].get("v", []),
                "av": results[video_id].get("av", []),
            }
        }

    out_path = f"../../supplimentary/models-results/{args.model}.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)


if __name__ == "__main__":
    main()
