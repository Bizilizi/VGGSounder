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
    id2original = pd.read_csv("supplimentary/data/vggsound/test.csv", names=["id", "class"])
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

                # add decision for original class, but only add decision for other
                # classes if they are True
                is_original = class_name == id2original[video_id]
                if is_original or decision:
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


def decide(
    df: pd.DataFrame, by: str, tie_breaker: str = None, verbose: bool = False
) -> pd.DataFrame:
    group_cols = [col for col in df.columns if col not in ["annotator", "value"]]

    def _majority_vote(group):
        """
        Decide the value of a group by majority vote.
        """
        # Convert values to numeric: True=1, False=-1, None/NaN=0
        values = group["value"].copy()
        values = values.replace({True: 1, False: -1})
        values = values.fillna(0)

        # Apply tie breaker weights if specified
        if tie_breaker is not None:
            tie_breaker_mask = group["annotator"] == tie_breaker
            values.loc[tie_breaker_mask & (values == 1)] = 1.5
            values.loc[tie_breaker_mask & (values == -1)] = -1.5

        return pd.Series({"value": values.sum() >= 0})

    def _dawid_skene(df, group_cols):
        """
        Decide the value of a group by Dawid-Skene.
        """
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

        ds = DawidSkene()
        result = ds.fit_predict(ds_df[["task", "worker", "label"]])
        decision_df["value"] = result.map({"True": True, "False": False})

        decision_df = decision_df.drop(columns=["worker", "label"])
        return decision_df

    if by == "majority":
        grouped = df.groupby(group_cols, dropna=False)
        decision_df = grouped.apply(_majority_vote).reset_index()
    elif by == "dawid-skene":
        decision_df = _dawid_skene(df, group_cols)
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


def get_target_df():
    target_df = pd.read_csv("supplimentary/data/vggsound/test.csv", names=["video_id", "label"])
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

    labels = pd.read_csv("supplimentary/data/vggsound/classes.csv")["display_name"].tolist()
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
