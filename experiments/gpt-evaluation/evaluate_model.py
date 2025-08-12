import argparse
import time
import pandas as pd
from tqdm.auto import tqdm, trange
import ast
from llm_backend import InferenceBackend

SYSTEM_PROMPT = """
You are an intelligent chatbot designed for evaluating the correctness of generative outputs for classification pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task: 

- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer.
- The correct answer, might contain multiple classes. Treat them independently and evaluate the correctness of all them w.r.t predicted answer.

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.

Please generate the response in the form of a Python dictionary string where names of classes are keys and values are dictionary strings  with keys ’pred’ and ’score’, where value of ’pred’ is a string of ’yes’ or ’no’ and value of ’score’ is in INTEGER, not STRING.

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: 

{"male speech, man speaking": {"pred": "yes", "score": 4}, "playing banjo": {"pred": "no", "score": 0}}

Example 1.

<Question> 
Identify the main sounds present in the given audio clip with a few words.

<Correct Answers> 
["cat caterwauling", "cat meowing"] 

<Predicted Answer> 
The main sounds present in the given audio clip are:

1. A ticking sound, possibly from a clock or timer.
2. A mechanical sound, which could be from a machine or device.
3. A human voice, which is speaking in the background.

Output: {"cat caterwauling": {"pred": "no", "score": 0}, "cat meowing": {"pred": "no", "score": 0}}

Example 2.

<Question> 
What actions are being performed in this audio, explain all sounds and actions in the audio? Please provide a short answer.

<Correct Answers> 
["cuckoo bird calling", "mynah bird singing", "bird chirping, tweeting"]

<Predicted Answer> 
The audio features a cuckoo bird calling in the distance and some chirping and tweeting from smaller birds.

Output: {"cuckoo bird calling": {"pred": "yes", "score": 5}, "mynah bird singing": {"pred": "no", "score": 0}, "bird chirping, tweeting": {"pred": "no", "score": 5}}

Example 3.

<Question> 
What actions are being performed in this video, explain all sounds and actions in the video? Please provide a short answer.

<Correct Answers> 
["male speech, man speaking", "playing hammond organ"]

<Predicted Answer> 
The video shows a man who is playing regular piano and speaking with someone.

Output: {"male speech, man speaking": {"pred": "yes", "score": 5}, "playing hammond organ": {"pred": "yes", "score": 3}}
"""

USER_PROMPT = """<Question>
{QUESTION}

<Correct Answers> 
{ANSWER}

<Predicted Answer>
{PREDICTION}
"""


def get_question(modality):
    if modality == "a":
        return "What actions are being performed in this audio, explain all sounds and actions in the audio? Please provide a short answer."
    else:
        return "What actions are being performed in this video, explain all sounds and actions in the video? Please provide a short answer."


def get_target(target_csv):
    # gorupby by video_id and then turn to list
    df = pd.read_csv(target_csv)[["video_id", "label"]]
    grouped = df.groupby("video_id")["label"].apply(list)
    grouped = grouped.reset_index()
    grouped["video_id"] = grouped["video_id"].astype(str) + ".mp4"

    return grouped


def evaluate_with_llm(llm, question, predictions, targets):
    batch = []
    for prediction, target in zip(predictions, targets):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    QUESTION=question, ANSWER=target, PREDICTION=prediction
                ),
            },
            {"role": "user", "content": "/no_think"},  # comment to enable thinking
        ]

        batch.append(messages)

    outputs = llm.batch_infer(batch)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B-Q4_K_M.gguf")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--window", type=str, default=None)
    parser.add_argument("--prediction_csv", type=str, default=None)
    parser.add_argument(
        "--target_csv",
        type=str,
        default="../data/crafted-labels/final_0.5_dawid-skene_heuristic_pivoted.csv",
    )
    parser.add_argument("--backend", type=str, default="transformers")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dev", type=bool, default=False)
    parser.add_argument("--cache", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rerun", type=bool, default=False)
    parser.add_argument("--make_copy", type=bool, default=False)

    args = parser.parse_args()

    print(f"Loading prediction csv: {args.prediction_csv}")
    prediction_df = pd.read_csv(args.prediction_csv)
    target_df = get_target(args.target_csv)

    merged_df = prediction_df.merge(target_df, on="video_id", how="inner")

    if "csv/av/" in args.prediction_csv:
        modality = "av"
    elif "csv/a/" in args.prediction_csv:
        modality = "a"
    elif "csv/v/" in args.prediction_csv:
        modality = "v"
    else:
        raise ValueError(f"Unknown modality for {args.prediction_csv}")

    question = get_question(modality)

    start_time = time.time()
    llm = InferenceBackend(
        args.model, backend_type=args.backend, cache_dir=args.cache, seed=args.seed
    )

    print(f"Model loading time: {time.time() - start_time:.2f} seconds")

    # skip already processed videos
    if args.rerun:
        merged_df["suggestions"] = ["[]"] * len(merged_df)
        start_idx = 0
    else:
        start_idx = merged_df[merged_df["suggestions"] == "[]"].index[0]

    if args.make_copy:
        args.prediction_csv = args.prediction_csv.replace(".csv", "_copy.csv")

    for idx in trange(start_idx, len(merged_df), args.batch_size, desc="Evaluating"):
        batch_df = merged_df.iloc[idx : idx + args.batch_size]

        batch_video_ids = batch_df["video_id"].tolist()
        batch_targets = batch_df["label"].tolist()
        batch_responses = batch_df["response"].tolist()

        outputs = evaluate_with_llm(llm, question, batch_responses, batch_targets)

        if not args.dev:
            merged_df.loc[
                merged_df.index[idx : idx + args.batch_size], "suggestions"
            ] = outputs
            merged_df[["video_id", "suggestions", "response"]].to_csv(
                args.prediction_csv, index=False
            )

        tqdm.write(f"Video {batch_video_ids} evaluated. Prediction: {outputs}")
