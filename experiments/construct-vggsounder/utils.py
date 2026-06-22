import glob
import json
import typing as t
from collections import defaultdict

import numpy as np
import pandas as pd
from crowdkit.aggregation import DawidSkene
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

META_CLASSES = ["background_music", "static_image", "voice_over"]

MUSIC_CLASSES = [
    "female singing",
    "playing bassoon",
    "playing piano",
    "playing harp",
    "male singing",
    "playing bass guitar",
    "playing violin, fiddle",
    "orchestra",
    "playing acoustic guitar",
    "playing cello",
    "playing electric guitar",
    "playing didgeridoo",
    "playing banjo",
    "playing flute",
    "playing accordion",
    "playing drum kit",
    "playing trombone",
    "playing saxophone",
    "playing cymbal",
    "playing marimba, xylophone",
    "singing bowl",
    "playing clarinet",
    "playing hammond organ",
    "playing tabla",
    "playing glockenspiel",
    "playing harpsichord",
    "playing theremin",
    "beat boxing",
    "child singing",
    "playing snare drum",
    "playing erhu",
    "yodelling",
    "playing bagpipes",
    "playing steel guitar, slide guitar",
    "playing bongo",
    "playing synthesizer",
    "playing ukulele",
    "playing cornet",
    "playing vibraphone",
    "people whistling",
    "playing french horn",
    "playing bass drum",
    "playing sitar",
    "playing electronic organ",
    "playing trumpet",
    "playing harmonica",
    "playing mandolin",
    "playing tambourine",
    "playing double bass",
    "tapping guitar",
    "singing choir",
    "rapping",
    "playing timpani",
    "playing timbales",
    "playing gong",
    "playing oboe",
    "playing zither",
    "playing steelpan",
    "playing guiro",
    "playing congas",
    "playing tympani",
    "playing bugle",
    "playing djembe",
    "people humming",
    "wind chime",
    "playing shofar",
    "playing tuning fork",
    "playing castanets",
]

SPEECH_CLASSES = [
    "female speech, woman speaking",
    "male speech, man speaking",
    "child speech, kid speaking",
    "people whispering",
]

# Mapping from class to list of classes that should be added
HEURISTIC = {
    "playing timpani": ["playing tympani"],
    "playing tympani": ["playing timpani"],
    "dog bow-wow": ["dog barking"],
    "dog barking": ["dog bow-wow"],
    "airplane flyby": ["airplane"],
    "barn swallow calling": ["bird chirping, tweeting"],
    "eagle screaming": ["bird squawking"],
    "canary calling": ["bird chirping, tweeting"],
    "mynah bird singing": ["bird chirping, tweeting"],
    "magpie calling": ["bird squawking"],
    "warbler chirping": ["bird chirping, tweeting"],
    "wood thrush calling": ["bird chirping, tweeting"],
    "goose honking": ["bird squawking"],
    "duck quacking": ["bird squawking"],
    "penguins braying": ["bird squawking"],
    "baltimore oriole calling": ["bird chirping, tweeting"],
    "crow cawing": ["bird squawking"],
    "baby babbling": ["people babbling"],
    "bull bellowing": ["cattle mooing"],
    "cow lowing": ["cattle mooing"],
    "people eating noodle": ["people eating"],
    "people eating apple": ["people eating"],
    "eating with cutlery": ["people eating"],
    "bathroom ventilation fan running": ["running electric fan"],
    "striking bowling": ["bowling impact"],
}


def get_gold_standard_df():
    """
    Creates a dataframe
        id | class | is_meta | is_original | modality | annotator | value
    from the goldstandard annotations.
    These annotations did not specifically include meta labels, so they have to be
    inferred from the classes.

    Returns:
        pd.DataFrame: A dataframe with the goldstandard annotations.
    """
    df = pd.read_csv("supplimentary/data/goldstandard/goldstandard-workers-results.csv")
    df = df.rename(columns={"label": "class", "original": "is_original"})
    df["is_meta"] = False

    old_labeler_cols = [col for col in df.columns if "labeler" in col]
    labeler_cols = [col.replace("labeler", "goldstandard") for col in old_labeler_cols]
    df = df.rename(
        columns={old: new for old, new in zip(old_labeler_cols, labeler_cols)}
    )

    meta_classes_by_labeler = []
    for id, group in df.groupby("id"):
        # Infer background music from music class decisions
        # Background music is any audible but invisible music
        music = group[group["class"].isin(MUSIC_CLASSES)]
        audible_music = (
            music[music["modality"] == "audible"][labeler_cols]
            .fillna(False)
            .astype(bool)
        ).reset_index(drop=True)
        visible_music = (
            music[music["modality"] == "visible"][labeler_cols]
            .fillna(False)
            .astype(bool)
        ).reset_index(drop=True)
        background_music = (audible_music & ~visible_music).any(axis=0).to_dict()
        meta_classes_by_labeler.append(
            {
                "id": id,
                "class": "background_music",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                **background_music,
            }
        )

        # Infer voice over from speech class decisions
        # Voice over is any audible but invisible speech
        speech_labels = group[group["class"].isin(SPEECH_CLASSES)]
        audible_speech = (
            speech_labels[speech_labels["modality"] == "audible"][labeler_cols]
            .fillna(False)
            .astype(bool)
        ).reset_index(drop=True)
        visible_speech = (
            speech_labels[speech_labels["modality"] == "visible"][labeler_cols]
            .fillna(False)
            .astype(bool)
        ).reset_index(drop=True)
        voice_over = (audible_speech & ~visible_speech).any(axis=0).to_dict()

        meta_classes_by_labeler.append(
            {
                "id": id,
                "class": "voice_over",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                **voice_over,
            }
        )

        # We can't infer the static image class from the labeled classes
        meta_classes_by_labeler.append(
            {
                "id": id,
                "class": "static_image",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                **{labeler: None for labeler in old_labeler_cols},
            }
        )
    meta_classes_df = pd.DataFrame(meta_classes_by_labeler)

    # concatenate classes and meta classes
    df = pd.concat([df, meta_classes_df])

    # unpivot labelers
    df = df.melt(
        id_vars=["id", "class", "is_meta", "is_original", "modality"],
        value_vars=labeler_cols,
        var_name="annotator",
        value_name="value",
    )

    return df


def get_inhouse_df():
    """
    Creates a dataframe
        id | class | is_meta | is_original | modality | annotator | value
    from the inhouse annotations.

    Returns:
        pd.DataFrame: A dataframe with the goldstandard annotations.
    """
    df = pd.read_csv("supplimentary/data/inhouse+meta.csv")
    df = df.rename(columns={"label": "class"})

    rows = []
    for _, row in df.iterrows():
        video_id = row["id"].split(".")[0]
        rows.append(
            {
                "id": video_id,
                "class": row["class"],
                "is_meta": False,
                "is_original": True,
                "modality": "audible",
                "annotator": "inhouse",
                "value": row["audible"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": row["class"],
                "is_meta": False,
                "is_original": True,
                "modality": "visible",
                "annotator": "inhouse",
                "value": row["visible"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "background_music",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": "inhouse",
                "value": row["background_music"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "voice_over",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": "inhouse",
                "value": row["voice_over"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "static_image",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": "inhouse",
                "value": row["static_images"],
            }
        )

    return pd.DataFrame(rows)


def get_mturk_original_meta_df():
    """
    Creates a dataframe
        id | class | is_meta | is_original | modality | annotator | value
    from the MTurk run with only original and meta labels.

    Returns:
        pd.DataFrame: A dataframe with the MTurk results for the original and meta labels.
    """
    df = pd.read_csv("supplimentary/mturk-annotations/original+meta.csv")
    rows = []
    for _, row in df.iterrows():
        video_id = row["Input.video_url"].split("/")[-1].split(".")[0]
        class_name = row["Input.label"]
        worker = row["WorkerId"]
        answers = json.loads(row["Answer.taskAnswers"])[0]
        rows.append(
            {
                "id": video_id,
                "class": class_name,
                "is_meta": False,
                "is_original": True,
                "modality": "audible",
                "annotator": worker,
                "value": answers["audible"]["True"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": class_name,
                "is_meta": False,
                "is_original": True,
                "modality": "visible",
                "annotator": worker,
                "value": answers["visible"]["True"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "background_music",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": worker,
                "value": answers["background_music"]["True"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "voice_over",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": worker,
                "value": answers["voice_over"]["True"],
            }
        )
        rows.append(
            {
                "id": video_id,
                "class": "static_image",
                "is_meta": True,
                "is_original": False,
                "modality": None,
                "annotator": worker,
                "value": answers["static_image"]["True"],
            }
        )
    return pd.DataFrame(rows)


def get_mturk_df(consider_hits_independent: bool = True):
    """
    Creates a dataframe
        id | class | is_meta | is_original | modality | annotator | value
    from the MTurk results from the first runs, which included original and meta labels,
    as well as additional labels.

    Returns:
        pd.DataFrame: A dataframe with the MTurk results from the first run.
    """
    # Collect all MTurk results
    csv_paths = glob.glob("supplimentary/mturk-annotations/*.csv")
    ignore_paths = ["all_batches_with_scores", "original+meta"]

    csv_files = []
    for csv_path in csv_paths:
        if any(ignore_path in csv_path for ignore_path in ignore_paths):
            continue
        df = pd.read_csv(csv_path)
        csv_files.append(df)
    df = pd.concat(csv_files)
    df.drop_duplicates(subset=["WorkerId", "HITId", "Answer.taskAnswers"], inplace=True)

    # Get the mapping from video id to original class
    id2original = pd.read_csv(
        "supplimentary/data/vggsound/test.csv", names=["id", "class"]
    )
    id2original["id"] = id2original["id"].str.replace(".mp4", "", regex=False)
    id2original = dict(zip(id2original["id"], id2original["class"]))

    rows = []
    unknown_ids = []
    for _, row in df.iterrows():
        if consider_hits_independent:
            worker = f"mturk_{row['WorkerId']}-{row['HITId']}"
        else:
            worker = row["WorkerId"]
        answers = json.loads(row["Answer.taskAnswers"])[0]

        for video_url, label_decisions in answers.items():
            if not video_url.startswith("https://") or isinstance(label_decisions, str):
                continue
            video_id = video_url.split("/")[-1].split(".")[0]

            # In this run, the proposals contained 7 ids that are not in the test set,
            # presumably from proposals generated by a model on a slightly different
            # version of the test set. We ignore these, but track them for now.
            if video_id not in id2original:
                unknown_ids.append(video_id)
                continue

            for class_name, decision in label_decisions.items():
                # add decisions for meta classes
                if class_name in META_CLASSES:
                    rows.append(
                        {
                            "id": video_id,
                            "class": class_name,
                            "is_meta": True,
                            "is_original": False,
                            "modality": None,
                            "annotator": worker,
                            "value": decision,
                        }
                    )
                    continue

                assert (
                    "audible" in class_name or "visible" in class_name
                ), f"Invalid class name: {class_name}"

                modality = "audible" if "audible" in class_name else "visible"
                class_name = class_name.replace(" audible", "").replace(" visible", "")

                # Store the worker's actual decision for EVERY shown proposal,
                # including explicit rejections (value=False). A label with no
                # row means the worker was never shown it (abstain), which is
                # distinct from an explicit "no".
                is_original = class_name == id2original[video_id]
                rows.append(
                    {
                        "id": video_id,
                        "class": class_name,
                        "is_meta": False,
                        "is_original": is_original,
                        "modality": modality,
                        "annotator": worker,
                        "value": decision,
                    }
                )
    # Print unknown ids for debugging, should be 7 ids.
    print(
        f"Encountered {len(set(unknown_ids))} unknown video ids {len(unknown_ids)} times:"
    )
    for unknown_id in set(unknown_ids):
        print(f"  {unknown_id}")
    return pd.DataFrame(rows)


def get_annotator_accuracy(annotations: pd.DataFrame, decision: pd.DataFrame):
    merge_cols = [col for col in decision.columns if col not in ["value"]]
    merged = pd.merge(
        annotations, decision, on=merge_cols, suffixes=("_annotator", "_decision")
    )

    scores = {}
    for annotator in tqdm(
        merged["annotator"].unique(), desc="Computing annotator accuracy"
    ):
        data = merged[merged["annotator"] == annotator]
        data = data.dropna(subset=["value_annotator", "value_decision"])

        if len(data) == 0:
            continue

        accuracy = (data["value_annotator"] == data["value_decision"]).mean()
        scores[annotator] = accuracy

    return scores


def get_annotator_f1(annotations: pd.DataFrame, decision: pd.DataFrame):
    # Get the label decisions per sample per annotator
    annotations = annotations.copy()
    annotations["class"] = annotations.apply(
        lambda x: f"{x['class']} {x['modality']}", axis=1
    )
    labels_pred = (
        annotations[annotations["value"]]
        .groupby(["id", "annotator"])
        .agg({"class": list})
        .reset_index()
    )

    decision = decision.copy()
    decision["class"] = decision.apply(
        lambda x: f"{x['class']} {x['modality']}", axis=1
    )
    labels_true = decision.groupby(["id"]).agg({"class": list}).reset_index()

    merged = pd.merge(labels_pred, labels_true, on="id", suffixes=("_pred", "_true"))

    scores = {}
    for annotator in tqdm(merged["annotator"].unique(), desc="Computing annotator F1"):
        data = merged[merged["annotator"] == annotator]
        data = data.dropna(subset=["class_pred", "class_true"])

        classes = set()
        for _, row in data.iterrows():
            classes.update(row["class_pred"])
            classes.update(row["class_true"])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        y_true = np.zeros((len(data), len(classes)))
        y_pred = np.zeros((len(data), len(classes)))

        for i, row in enumerate(data.itertuples()):
            for cls in row.class_true:
                y_true[i, class_to_idx[cls]] = 1
            for cls in row.class_pred:
                y_pred[i, class_to_idx[cls]] = 1

        # Get the F1 score
        f1 = f1_score(y_true, y_pred, average="micro")
        scores[annotator] = f1

    return scores


# Default reliability for annotators without a goldstandard-derived score
# (e.g. inhouse / goldstandard workers, whom we trust by construction).
DEFAULT_RELIABILITY = 1.0
_WEIGHT_EPS = 1e-6


def _clip_unit(x: float) -> float:
    """Clip a value to the open interval (0, 1) to keep logs/divisions finite."""
    return min(max(float(x), _WEIGHT_EPS), 1.0 - _WEIGHT_EPS)


def _vote_state(value) -> str:
    """Map a raw vote value to one of: 'true', 'false', 'na'."""
    if pd.isna(value):
        return "na"
    return "true" if bool(value) else "false"


def _atom_confusion(annotations: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    """Per-annotator atom-level confusion counts (tp, fp, fn, tn) against `decision`.

    Only EXPLICIT votes are scored: each annotator row with a non-null value is
    matched (inner join) against the decision atoms. Absent votes mean the
    annotator was never shown the proposal (abstain) and contribute nothing, so
    precision/recall/specificity reflect genuine decisions rather than
    reconstructed negatives.

    Returns:
        pd.DataFrame indexed by `annotator` with integer columns tp, fp, fn, tn.
    """
    merge_cols = [col for col in decision.columns if col != "value"]

    truth = decision.dropna(subset=["value"]).copy()
    truth["truth"] = truth["value"].astype(bool)
    truth = truth[merge_cols + ["truth"]]

    ann = annotations.dropna(subset=["value"]).copy()
    ann["pred"] = ann["value"].astype(bool)
    ann = ann[merge_cols + ["annotator", "pred"]].drop_duplicates(
        merge_cols + ["annotator"]
    )

    # Keep only atoms where the annotator explicitly voted AND a decision exists
    grid = ann.merge(truth, on=merge_cols, how="inner")

    grid["tp"] = grid["truth"] & grid["pred"]
    grid["fp"] = (~grid["truth"]) & grid["pred"]
    grid["fn"] = grid["truth"] & (~grid["pred"])
    grid["tn"] = (~grid["truth"]) & (~grid["pred"])

    return grid.groupby("annotator")[["tp", "fp", "fn", "tn"]].sum()


def get_annotator_prf(annotations: pd.DataFrame, decision: pd.DataFrame) -> dict:
    """Per-annotator micro precision/recall/f1 against `decision` (atom-based)."""
    counts = _atom_confusion(annotations, decision)
    scores = {}
    for annotator, row in counts.iterrows():
        tp, fp, fn = float(row["tp"]), float(row["fp"]), float(row["fn"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores[annotator] = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def get_annotator_confusion(annotations: pd.DataFrame, decision: pd.DataFrame) -> dict:
    """Per-annotator sensitivity (tpr) and specificity (tnr) against `decision`."""
    counts = _atom_confusion(annotations, decision)
    scores = {}
    for annotator, row in counts.iterrows():
        tp, fp, fn, tn = (
            float(row["tp"]),
            float(row["fp"]),
            float(row["fn"]),
            float(row["tn"]),
        )
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        scores[annotator] = {"tpr": tpr, "tnr": tnr}
    return scores


def get_annotator_confusion_counts(
    annotations: pd.DataFrame, decision: pd.DataFrame
) -> pd.DataFrame:
    """Per-worker error-count matrices in crowdkit's expected format.

    The returned frame is indexed by (`worker`, `label`) where `label` is the
    OBSERVED label, with columns for each TRUE label, so that
    `result.loc[worker, observed, true]` is the historical count that `worker`
    produced `observed` given the true label was `true`. Suitable for
    `DawidSkene(initial_error_strategy="addition").fit_predict(initial_error=...)`.
    """
    counts = _atom_confusion(annotations, decision)
    index = []
    rows = []
    for worker, row in counts.iterrows():
        tp, fp, fn, tn = (
            float(row["tp"]),
            float(row["fp"]),
            float(row["fn"]),
            float(row["tn"]),
        )
        # observed True given true {True, False}
        index.append((worker, "True"))
        rows.append({"True": tp, "False": fp})
        # observed False given true {True, False}
        index.append((worker, "False"))
        rows.append({"True": fn, "False": tn})

    multi_index = pd.MultiIndex.from_tuples(index, names=["worker", "label"])
    return pd.DataFrame(rows, index=multi_index)[["True", "False"]].astype(float)


def decide(
    df: pd.DataFrame,
    by: str,
    tie_breaker: str = None,
    verbose: bool = False,
    count_missing_as_false: bool = True,
    annotator_weights: dict = None,
    algo_params: dict = None,
    ds_initial_error: pd.DataFrame = None,
    ds_true_labels: pd.DataFrame = None,
) -> pd.DataFrame:
    group_cols = [col for col in df.columns if col not in ["annotator", "value"]]

    params = {
        "alpha": 0.3,  # soft-negative: weight of an EXPLICIT no (vs +1 for a yes)
        "tau_high": 0.7,  # expert-override: f1 threshold to be an "expert"
        "tau_low": 0.5,  # expert-override: recall threshold below which an explicit no is overridable
        "prior": 0.5,  # bayesian-logodds: prior probability a label is present
        "weight_key": "f1",  # reliability-weighted: which score to use as weight
        "inhouse_boost": 1.0,  # soft-negative: multiplier on inhouse votes (gen-3 boost)
    }
    if algo_params:
        params.update(algo_params)
    annotator_weights = annotator_weights or {}

    # Explicit-negative semantics: a stored row is a genuine vote (True / False),
    # an absent row means the annotator was never shown the proposal (abstain).
    # No missing-vote reconstruction is performed except for the legacy
    # `majority` + count_missing_as_false=True path.
    annotators_by_id = None
    id_group_pos = None
    if by == "majority" and count_missing_as_false:
        if "id" not in df.columns:
            raise ValueError("count_missing_as_false requires an 'id' column")
        annotators_by_id = df.groupby("id")["annotator"].agg(lambda s: set(s)).to_dict()
        id_group_pos = group_cols.index("id")

    def _video_id(group):
        key = group.name
        return key[id_group_pos] if isinstance(key, tuple) else key

    def _vote_map(group):
        return dict(zip(group["annotator"], group["value"]))

    def _w(annotator, key):
        return annotator_weights.get(annotator, {}).get(key, DEFAULT_RELIABILITY)

    def _counts(votes):
        n_true = sum(1 for v in votes.values() if _vote_state(v) == "true")
        n_false = sum(1 for v in votes.values() if _vote_state(v) == "false")
        return n_true, n_false

    def _majority_vote(group):
        """Decide a group by majority over explicit votes (abstain = 0)."""
        values = group["value"].copy()
        values = values.replace({True: 1, False: -1})
        values = values.fillna(0)

        if tie_breaker is not None:
            tie_breaker_mask = group["annotator"] == tie_breaker
            values.loc[tie_breaker_mask & (values == 1)] = 1.5
            values.loc[tie_breaker_mask & (values == -1)] = -1.5

        score = values.sum()
        if count_missing_as_false:
            expected_annotators = len(annotators_by_id[_video_id(group)])
            present_annotators = group["annotator"].nunique()
            score -= expected_annotators - present_annotators

        return pd.Series({"value": score > 0})

    def _soft_negative_vote(group):
        """Re-purposed Option 1: an explicit no counts -alpha (a rejection is
        weaker evidence than a selection); abstain contributes nothing.
        Inhouse votes are scaled by `inhouse_boost` (1.0 = no boost)."""
        votes = _vote_map(group)
        score = 0.0
        for annotator, value in votes.items():
            weight = params["inhouse_boost"] if annotator == "inhouse" else 1.0
            state = _vote_state(value)
            if state == "true":
                score += weight
            elif state == "false":
                score -= weight * params["alpha"]
        return pd.Series({"value": score > 0})

    def _reliability_weighted_vote(group):
        """Option 2: explicit votes weighted by annotator reliability (f1)."""
        votes = _vote_map(group)
        key = params["weight_key"]
        score = 0.0
        for annotator, value in votes.items():
            state = _vote_state(value)
            if state == "true":
                score += _w(annotator, key)
            elif state == "false":
                score -= _w(annotator, key)
        return pd.Series({"value": score > 0})

    def _precision_recall_vote(group):
        """Option 3: explicit 'yes' weighted by precision, explicit 'no' by recall."""
        votes = _vote_map(group)
        score = 0.0
        for annotator, value in votes.items():
            state = _vote_state(value)
            if state == "true":
                score += _w(annotator, "precision")
            elif state == "false":
                score -= _w(annotator, "recall")
        return pd.Series({"value": score > 0})

    def _expert_override_vote(group):
        """Option 4: majority over explicit votes, but a high-f1 'yes' overrides
        explicit no's that all come from low-recall workers."""
        votes = _vote_map(group)
        n_true, n_false = _counts(votes)

        base = (n_true - n_false) > 0
        if base:
            return pd.Series({"value": True})

        has_expert_true = any(
            _vote_state(value) == "true" and _w(annotator, "f1") >= params["tau_high"]
            for annotator, value in votes.items()
        )
        opposition_only_low_recall = n_false > 0 and all(
            _w(annotator, "recall") < params["tau_low"]
            for annotator, value in votes.items()
            if _vote_state(value) == "false"
        )
        return pd.Series({"value": has_expert_true and opposition_only_low_recall})

    def _bayesian_vote(group):
        """Option 5: Naive-Bayes log-odds over explicit votes only (abstain excluded)."""
        votes = _vote_map(group)
        prior = _clip_unit(params["prior"])
        logodds = np.log(prior / (1.0 - prior))
        for annotator, value in votes.items():
            state = _vote_state(value)
            tpr = _clip_unit(_w(annotator, "tpr"))
            tnr = _clip_unit(_w(annotator, "tnr"))
            if state == "true":
                logodds += np.log(tpr / (1.0 - tnr))
            elif state == "false":
                logodds += np.log((1.0 - tpr) / tnr)
            # 'na' contributes nothing
        return pd.Series({"value": logodds > 0})

    def _dawid_skene(df, group_cols, initial_error=None, true_labels_df=None):
        """Decide groups by Dawid-Skene, optionally seeded from the goldstandard."""
        ds_df = df.copy()
        # Consider each video-class-modality combination as a task
        ds_df["task"] = df.groupby(group_cols, dropna=False).ngroup()
        # Rename columns for Dawid-Skene
        ds_df = ds_df.rename(columns={"annotator": "worker", "value": "label"})
        # Convert boolean values to string labels for Dawid-Skene
        ds_df = ds_df.dropna(subset=["label"]).reset_index(drop=True)
        ds_df["label"] = ds_df["label"].map({True: "True", False: "False"})

        # Now keep only one row per task
        decision_df = ds_df.groupby("task").first().reset_index(drop=True)

        # Optional gold seeding: anchor known tasks to their true labels
        true_labels = None
        if true_labels_df is not None:
            key_to_task = ds_df.drop_duplicates(group_cols)[group_cols + ["task"]]
            tl = true_labels_df.dropna(subset=["value"]).merge(
                key_to_task, on=group_cols, how="inner"
            )
            tl["ds_label"] = tl["value"].map({True: "True", False: "False"})
            tl = tl.drop_duplicates("task")
            true_labels = pd.Series(tl["ds_label"].values, index=tl["task"].values)

        strategy = "addition" if initial_error is not None else None
        ds = DawidSkene(initial_error_strategy=strategy)
        result = ds.fit_predict(
            ds_df[["task", "worker", "label"]],
            true_labels=true_labels,
            initial_error=initial_error,
        )
        decision_df["value"] = result.map({"True": True, "False": False})

        decision_df = decision_df.drop(columns=["worker", "label"])
        return decision_df

    vote_funcs = {
        "majority": _majority_vote,
        "soft-negative": _soft_negative_vote,
        "reliability-weighted": _reliability_weighted_vote,
        "precision-recall": _precision_recall_vote,
        "expert-override": _expert_override_vote,
        "bayesian-logodds": _bayesian_vote,
    }

    if by in vote_funcs:
        grouped = df.groupby(group_cols, dropna=False)
        decision_df = grouped.apply(vote_funcs[by]).reset_index()
    elif by == "dawid-skene":
        decision_df = _dawid_skene(df, group_cols)
    elif by == "dawid-skene-seeded":
        decision_df = _dawid_skene(
            df,
            group_cols,
            initial_error=ds_initial_error,
            true_labels_df=ds_true_labels,
        )
    else:
        raise ValueError(f"Invalid decision method: {by}")

    if verbose:
        annotator2score = get_annotator_accuracy(df, decision_df)
        avg_score = sum(annotator2score.values()) / len(annotator2score)
        print(f"Average agreement: {avg_score:.3f}")
        ours = [
            "inhouse",
            "goldstandard_0",
            "goldstandard_1",
            "goldstandard_2",
            "goldstandard_3",
        ]
        for _ours in ours:
            if not _ours in annotator2score:
                continue
            print(f"    {_ours}: {annotator2score.get(_ours, 0):.3f}")
        # for annotator, score in annotator2score.items():
        #     print(f"    {annotator}: {score:.3f}")

    return decision_df


def apply_inhouse_override(
    decision_df: pd.DataFrame, inhouse_df: pd.DataFrame
) -> pd.DataFrame:
    """Hard-override merged decisions with inhouse annotations (gen-3 variant B).

    Inhouse covers only the original label's audible/visible atoms plus the 3
    meta labels. Wherever inhouse cast an explicit (non-null) vote on an atom,
    the merged decision's `value` is replaced by the inhouse value. Must be
    applied AFTER decide() and BEFORE the value==True filter so an inhouse "no"
    deletes a merged label and an inhouse "yes" restores one.

    Args:
        decision_df: output of decide(); one row per atom with a `value` column.
        inhouse_df: output of get_inhouse_df(); per-atom inhouse votes.

    Returns:
        pd.DataFrame: decision_df with overridden `value`s (same shape/columns).
    """
    key_cols = [
        col
        for col in ["id", "class", "modality", "is_meta"]
        if col in decision_df.columns and col in inhouse_df.columns
    ]
    overrides = (
        inhouse_df.dropna(subset=["value"])[key_cols + ["value"]]
        .drop_duplicates(key_cols, keep="last")
        .rename(columns={"value": "inhouse_value"})
    )
    out = decision_df.merge(overrides, on=key_cols, how="left")
    overridden = out["inhouse_value"].notna()
    out.loc[overridden, "value"] = out.loc[overridden, "inhouse_value"].astype(bool)
    out = out.drop(columns=["inhouse_value"])
    changed = int(
        (
            decision_df["value"].reset_index(drop=True)
            != out["value"].reset_index(drop=True)
        )[overridden.reset_index(drop=True)].sum()
    )
    print(
        f"[inhouse-override] inhouse covers {int(overridden.sum())} / {len(out)} "
        f"decision atoms; changed {changed} values"
    )
    return out


def get_target_df():
    target_df = pd.read_csv(
        "supplimentary/data/vggsound/test.csv", names=["video_id", "label"]
    )
    target_df["video_id"] = target_df["video_id"].str.split(".").str[0]
    return target_df


class VGGLabelEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, labels: t.List[str]):
        self.label_encoder.fit(labels)

    def transform(self, labels: t.List[str]) -> t.List[int]:
        indices = self.label_encoder.transform(labels)
        one_hot = np.zeros(self.label_encoder.classes_.shape[0])

        if len(indices) > 0:
            one_hot[indices] = 1

        return one_hot


def get_class_encoder() -> VGGLabelEncoder:
    label_encoder = VGGLabelEncoder()

    labels = pd.read_csv("supplimentary/data/vggsound/classes.csv")[
        "display_name"
    ].tolist()
    a_labels = [f"{label} audible" for label in labels]
    v_labels = [f"{label} visible" for label in labels]

    av_labels = a_labels + v_labels

    label_encoder.fit(av_labels)

    return label_encoder


def get_gold_standard() -> t.Dict[str, np.ndarray]:
    gold_standard = defaultdict(list)
    df = pd.read_csv("supplimentary/data/goldstandard/goldstandard.csv")

    for _, row in df.iterrows():

        labels = []
        label = row["label"]
        modality = row["modality"]

        if "A" in modality:
            labels.append(f"{label} audible")
        elif "V" in modality:
            labels.append(f"{label} visible")

        if labels:
            gold_standard[row["id"]].extend(labels)

    return gold_standard


def drop_extra_classes(labels: t.List[str]) -> t.List[str]:
    return [label for label in labels if label not in META_CLASSES]


def evaluate_worker_performance(
    results: t.Dict[str, t.List[str]],
    gold_standard: t.Dict[str, t.List[str]],
    encoder: VGGLabelEncoder,
) -> float:

    gold_standart_subbmitions = []
    gold_standart_targets = []

    # select only gold standart subbmitions
    for video_url, labels in results.items():
        video_id = video_url.split("/")[-1].split(".")[0]
        if video_id in gold_standard and len(labels) > 0:
            labels = drop_extra_classes(labels)

            gold_standart_subbmitions.append(encoder.transform(labels))
            gold_standart_targets.append(encoder.transform(gold_standard[video_id]))

    # convert gold standart subbmitions to one-hot encoding
    # compare with gold standart targets
    gold_standart_subbmitions = np.array(gold_standart_subbmitions)
    gold_standart_targets = np.array(gold_standart_targets)

    precision = precision_score(
        gold_standart_targets,
        gold_standart_subbmitions,
        average="micro",
        zero_division=0,
    )
    recall = recall_score(
        gold_standart_targets,
        gold_standart_subbmitions,
        average="micro",
        zero_division=0,
    )
    score = f1_score(
        gold_standart_targets,
        gold_standart_subbmitions,
        average="micro",
        zero_division=0,
    )

    # convert numpy float to float
    return score, precision, recall


def keep_only(
    *, modality: str, video_labels: t.Dict[str, t.List[str]]
) -> t.Dict[str, t.List[str]]:
    return {
        video_id: list(
            {
                label_without_modality(label)
                for label in labels
                if label_modality(label) in modality
            }
        )
        for video_id, labels in video_labels.items()
    }


def label_without_modality(label: str) -> str:
    return label.replace(" audible", "").replace(" visible", "")


def label_modality(label: str) -> str:
    if " audible" in label:
        return "A"
    elif " visible" in label:
        return "V"
    else:
        return ""


def append_modality(label: str, modality: str) -> str:
    if modality == "A":
        return f"{label} audible"
    elif modality == "V":
        return f"{label} visible"
    else:
        return label


def inject_heuristic_labels(labels: t.List[str]) -> t.List[str]:
    new_labels = []
    for label in labels:
        new_labels.append(label)

        # add extra classes based on heuristics
        original_label = label_without_modality(label)
        modality = label_modality(label)

        # fix tympani/timpani typo
        if original_label == "playing timpani":
            new_labels.append(append_modality("playing tympani", modality))

        if original_label == "playing tympani":
            new_labels.append(append_modality("playing timpani", modality))

        # fix dog bow-wow <-> dog barking
        if original_label == "dog bow-wow":
            new_labels.append(append_modality("dog barking", modality))

        if original_label == "dog barking":
            new_labels.append(append_modality("dog bow-wow", modality))

        # fix airplane flyby -> airplane
        if original_label == "airplane flyby":
            new_labels.append(append_modality("airplane", modality))

        # fix barn swallow calling -> bird chirping, tweeting
        if original_label == "barn swallow calling":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix eagle screaming -> bird squawking
        if original_label == "eagle screaming":
            new_labels.append(append_modality("bird squawking", modality))

        # fix canary calling -> bird chirping, tweeting
        if original_label == "canary calling":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix mynah bird singing -> bird chirping, tweeting
        if original_label == "mynah bird singing":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix magpie calling -> bird squawking
        if original_label == "magpie calling":
            new_labels.append(append_modality("bird squawking", modality))

        # fix warbler chirping -> bird chirping, tweeting
        if original_label == "warbler chirping":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix wood thrush calling -> bird chirping, tweeting
        if original_label == "wood thrush calling":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix goose honking -> bird squawking
        if original_label == "goose honking":
            new_labels.append(append_modality("bird squawking", modality))

        # fix duck quacking -> bird squawking
        if original_label == "duck quacking":
            new_labels.append(append_modality("bird squawking", modality))

        # fix penguins braying -> bird squawking
        if original_label == "penguins braying":
            new_labels.append(append_modality("bird squawking", modality))

        # fix baltimore oriole calling -> bird chirping, tweeting
        if original_label == "baltimore oriole calling":
            new_labels.append(append_modality("bird chirping, tweeting", modality))

        # fix crow cawing -> bird squawking
        if original_label == "crow cawing":
            new_labels.append(append_modality("bird squawking", modality))

        # fix baby babbling -> people babbling
        if original_label == "baby babbling":
            new_labels.append(append_modality("people babbling", modality))

        # fix bull bellowing -> cattle mooing
        if original_label == "bull bellowing":
            new_labels.append(append_modality("cattle mooing", modality))

        # fix cow lowing -> cattle mooing
        if original_label == "cow lowing":
            new_labels.append(append_modality("cattle mooing", modality))

        # fix people eating noodle -> people eating
        if original_label == "people eating noodle":
            new_labels.append(append_modality("people eating", modality))

        # fix people eating apple -> people eating
        if original_label == "people eating apple":
            new_labels.append(append_modality("people eating", modality))

        # fix eating with cutlery -> people eating
        if original_label == "eating with cutlery":
            new_labels.append(append_modality("people eating", modality))

        # fix bathroom ventilation fan running -> running electric fan
        if original_label == "bathroom ventilation fan running":
            new_labels.append(append_modality("running electric fan", modality))

        # fix striking bowling -> bowling impact
        if original_label == "striking bowling":
            new_labels.append(append_modality("bowling impact", modality))

    return new_labels


def safely_inject_target(labels, video_id, target_df):
    target = target_df[target_df["video_id"] == video_id]["label"].values[0]

    if target in {label_without_modality(label) for label in labels}:
        return labels
    else:
        return labels + [append_modality(target, "A"), append_modality(target, "V")]
