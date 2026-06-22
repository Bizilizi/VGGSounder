# %%
import os
from pathlib import Path

import utils
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

pd.set_option("future.no_silent_downcasting", True)


"""
Available options:
- bayesian-logodds
- dawid-skene
- dawid-skene-seeded ❤️‍🔥
- expert-override
- majority
- precision-recall
- reliability-weighted
- soft-negative
"""
DECIDE_BY = "dawid-skene-seeded"

"""
Available options:
- inhouse-override ❤️‍🔥
- inhouse-boost
"""
VARIANT = "inhouse-override"

# Algorithm parameters (defaults live in utils.decide; override here if desired).
ALGO_PARAMS = {}
TIE_BREAKER = None
if VARIANT == "inhouse-boost":
    # majority: inhouse votes count +-1.5; soft-negative: same boost factor.
    ALGO_PARAMS["inhouse_boost"] = 1.5
    TIE_BREAKER = "inhouse"

# Reliability tables, filled in once the goldstandard decision is available.
ANNOTATOR_WEIGHTS = {}
DS_INITIAL_ERROR = None
DS_TRUE_LABELS = None


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except NameError:
        return False


IN_JUPYTER = is_notebook()
if not IN_JUPYTER:
    display = print

# CORRECT ORIGINAL LABELS and ADD META LABELS
# %%
goldstandard = utils.get_gold_standard_df()
display(goldstandard.head())

# %%
inhouse = utils.get_inhouse_df()
display(inhouse.head())

# %%
# Preliminary decisions based on goldstandard and inhouse labels
goldstandard_inhouse = pd.concat([goldstandard, inhouse])
goldstandard_inhouse["is_goldset"] = goldstandard_inhouse["id"].isin(goldstandard["id"])
goldstandard_inhouse_goldset = goldstandard_inhouse[
    goldstandard_inhouse["is_goldset"]
].drop(columns=["is_goldset"])
goldstandard_inhouse_nongoldset = goldstandard_inhouse[
    ~goldstandard_inhouse["is_goldset"]
].drop(columns=["is_goldset"])

# On the goldset, merge decisions from 5 annotations
preliminary_goldset = utils.decide(
    goldstandard_inhouse_goldset, by="dawid-skene", verbose=True
)
preliminary_goldset_OM = preliminary_goldset[
    preliminary_goldset["is_original"] | preliminary_goldset["is_meta"]
]
preliminary_goldset_other = preliminary_goldset[
    ~preliminary_goldset["is_original"] & ~preliminary_goldset["is_meta"]
]

# Everything else has a single annotation (inhouse without goldstandard IDs)
preliminary_nongoldset_OM = goldstandard_inhouse_nongoldset.drop(columns=["annotator"])

# %%
# Format results from MTurk runs
mturk_om_run = utils.get_mturk_original_meta_df()
mturk_first_run = utils.get_mturk_df(consider_hits_independent=False)
mturk = pd.concat([mturk_om_run, mturk_first_run])
display(mturk.head())

# %%
# Grade all annotators (accuracy) for blacklisting / top-3 selection
annotator_scores_goldset_OM = utils.get_annotator_accuracy(
    mturk, preliminary_goldset_OM
)
annotator_scores_nongoldset_OM = utils.get_annotator_accuracy(
    mturk, preliminary_nongoldset_OM
)
annotator_scores_goldset_other_acc = utils.get_annotator_accuracy(
    mturk, preliminary_goldset_other
)

# %%
# Reliability tables used by the weighted / bayesian / seeded-DS algorithms.
# Computed against the trusted goldset decision (precision/recall/f1/tpr/tnr).
prf = utils.get_annotator_prf(mturk, preliminary_goldset)
conf = utils.get_annotator_confusion(mturk, preliminary_goldset)
ANNOTATOR_WEIGHTS = {
    annotator: {**prf.get(annotator, {}), **conf.get(annotator, {})}
    for annotator in set(prf) | set(conf)
}
DS_INITIAL_ERROR = utils.get_annotator_confusion_counts(mturk, preliminary_goldset)
DS_TRUE_LABELS = preliminary_goldset

if VARIANT == "inhouse-boost":
    # dawid-skene-seeded: strong diagonal prior counts so EM starts trusting
    # inhouse (it has no MTurk-derived confusion history).
    INHOUSE_PRIOR_COUNT = 50.0
    inhouse_prior = pd.DataFrame(
        [
            {"True": INHOUSE_PRIOR_COUNT, "False": 0.0},  # observed True
            {"True": 0.0, "False": INHOUSE_PRIOR_COUNT},  # observed False
        ],
        index=pd.MultiIndex.from_tuples(
            [("inhouse", "True"), ("inhouse", "False")], names=["worker", "label"]
        ),
    )
    DS_INITIAL_ERROR = pd.concat([DS_INITIAL_ERROR, inhouse_prior])

# %%
# Blacklist annotators with agreement below a threshold
threshold = 0.4
annotators = (
    set(annotator_scores_goldset_OM.keys())
    | set(annotator_scores_nongoldset_OM.keys())
    | set(annotator_scores_goldset_other_acc.keys())
)
blacklist = [
    annotator
    for annotator in annotators
    if annotator_scores_goldset_OM.get(annotator, 1) < threshold
    or annotator_scores_nongoldset_OM.get(annotator, 1) < threshold
    or annotator_scores_goldset_other_acc.get(annotator, 1) < threshold
]
print(f"Ignoring {len(blacklist)} ({len(blacklist) / len(annotators):.2%}) annotators")

# %%
# Merge good annotators with our labels, then keep only top-3 labelers per video.
# COMMON GEN-3 CHANGE: `inhouse` has no MTurk goldset score (defaults to 0) and
# used to be silently dropped here on 99.2% of videos; it is now ALWAYS kept
# alongside the top-3 scored workers (videos get up to 4 annotators).
merged = pd.concat([goldstandard, inhouse, mturk[~mturk["annotator"].isin(blacklist)]])

top_labelers_only = merged.copy()
labels = []
inhouse_kept = 0
for id, group in tqdm(top_labelers_only.groupby("id")):
    video_labelers = group["annotator"].unique()
    labelers_scores = {
        annotator: annotator_scores_goldset_OM.get(annotator, 0)
        for annotator in video_labelers
    }
    top_labelers = set(
        sorted(labelers_scores, key=labelers_scores.get, reverse=True)[:3]
    )
    if "inhouse" in video_labelers:
        top_labelers.add("inhouse")
        inhouse_kept += 1
    labels.append(group[group["annotator"].isin(top_labelers)])
top_labelers_only = pd.concat(labels)

annotators_per_video = top_labelers_only.copy()
annotators_per_video.drop_duplicates(["id", "annotator"], inplace=True)
annotators_per_video = annotators_per_video.groupby("id").count()["annotator"]
n_videos = annotators_per_video.shape[0]
print(
    f"Avg. annotators per video: {annotators_per_video.mean():.2f}, "
    f"std: {annotators_per_video.std():.2f}, min: {annotators_per_video.min()}, "
    f"max: {annotators_per_video.max()}"
)
print(
    f"Inhouse kept on {inhouse_kept} / {n_videos} videos "
    f"({inhouse_kept / n_videos:.2%})"
)
merged = top_labelers_only.copy()


# %%
def get_final_labels(
    merged, add_heuristics=True, labler_merging=True, decide_by="majority"
):
    if labler_merging:
        final = utils.decide(
            merged,
            by=decide_by,
            tie_breaker=TIE_BREAKER,  # only used by 'majority' (inhouse-boost)
            count_missing_as_false=False,  # absent vote = abstain (not shown)
            annotator_weights=ANNOTATOR_WEIGHTS,
            algo_params=ALGO_PARAMS,
            ds_initial_error=DS_INITIAL_ERROR,
            ds_true_labels=DS_TRUE_LABELS,
        )
        if VARIANT == "inhouse-override":
            # Hard-override BEFORE the value==True filter so an inhouse "no"
            # deletes a merged label and an inhouse "yes" restores one.
            final = utils.apply_inhouse_override(final, inhouse)
        final = final.dropna(subset=["value"])  # drop rows with no values
        final = final[final["value"]]  # keep only records of labels we keep
    else:
        final = merged.copy()
        print(f"Final shape: {final.shape}")

    # Add labels via heuristic
    if add_heuristics:
        add_labels = []
        for _, row in final[
            ~final["is_meta"] & final["class"].isin(utils.HEURISTIC.keys())
        ].iterrows():
            for add_class in utils.HEURISTIC[row["class"]]:
                add_row = row.copy()
                add_row["class"] = add_class
                add_row["is_heuristic"] = True
                add_labels.append(add_row)
        final.insert(4, "is_heuristic", False)
        final = pd.concat([final, pd.DataFrame(add_labels)])

    display(final.head())

    avg_meta_labels = final[final["is_meta"]].groupby("id").size().mean()
    print(f"Avg. meta labels per video: {avg_meta_labels:.2f}")
    avg_labels = final[~final["is_meta"]].groupby("id").size().mean()
    print(f"Avg. non-meta labels per video: {avg_labels:.2f}")

    return final


# %%
def get_pivoted_labels(final):
    final_grouped = final.groupby(["id"])

    rows = []
    for video_id, group in final_grouped:
        row_meta = {}
        meta_classes = group[group["is_meta"] & group["value"]]["class"].tolist()
        for meta_class in utils.META_CLASSES:
            row_meta[meta_class] = meta_class in meta_classes

        other_classes = group[~group["is_meta"] & group["value"]][
            ["class", "modality"]
        ].drop_duplicates()
        for class_name, group in other_classes.groupby("class"):
            row = {"video_id": video_id[0], "label": class_name}

            modalities = group["modality"].tolist()
            modality = ""
            if "audible" in modalities:
                modality += "A"
            if "visible" in modalities:
                modality += "V"
            row["modality"] = modality

            row.update(row_meta)
            rows.append(row)

    final_pivoted = pd.DataFrame(rows)
    display(final_pivoted.head())
    return final_pivoted


# %%
# ----------------------------------------------------------------------------- #
# Pivoted-format helpers used by the gen-4 final chain.
# ----------------------------------------------------------------------------- #
def modality_union(*mods):
    """Union of 'A'/'V'/'AV' modality strings -> ordered 'A'/'V'/'AV'."""
    chars = set()
    for m in mods:
        for c in str(m):
            if c in ("A", "V"):
                chars.add(c)
    return ("A" if "A" in chars else "") + ("V" if "V" in chars else "")


def _meta_lookup(video_id, meta_by_video):
    if video_id in meta_by_video.index:
        return meta_by_video.loc[video_id].to_dict()
    return {m: False for m in utils.META_CLASSES}


def apply_reverts(pivoted, verdicts, meta_by_video):
    """Revert every Gemini exists==True (video, label, modality) into `pivoted`.

    Unions the reverted modality onto an existing (video, label) row, or inserts
    a new row (meta taken from the video's existing meta, else all-False).
    Reverts win over may-2026 deletions.
    """
    acc = {}
    for _, r in pivoted.iterrows():
        acc[(r["video_id"], r["label"])] = {
            "modality": r["modality"],
            **{m: r[m] for m in utils.META_CLASSES},
        }
    for _, r in verdicts.iterrows():
        key = (r["video_id"], r["label"])
        if key in acc:
            acc[key]["modality"] = modality_union(acc[key]["modality"], r["modality"])
        else:
            meta = _meta_lookup(r["video_id"], meta_by_video)
            acc[key] = {"modality": r["modality"], **meta}
    rows = [
        {
            "video_id": k[0],
            "label": k[1],
            "modality": v["modality"],
            **{m: v[m] for m in utils.META_CLASSES},
        }
        for k, v in acc.items()
    ]
    return pd.DataFrame(rows, columns=pivoted.columns)


def apply_heuristics_pivoted(pivoted):
    """Apply utils.HEURISTIC label expansion on the pivoted dataset (heuristics last).

    For each row whose label is a heuristic key, add the mapped label(s) with the
    same modality + meta + video, then dedup on (video, label) unioning modality.
    """
    acc = {}

    def add(video_id, label, modality, meta):
        key = (video_id, label)
        if key in acc:
            acc[key]["modality"] = modality_union(acc[key]["modality"], modality)
        else:
            acc[key] = {"modality": modality, **meta}

    for _, r in pivoted.iterrows():
        meta = {m: r[m] for m in utils.META_CLASSES}
        add(r["video_id"], r["label"], r["modality"], meta)
        if r["label"] in utils.HEURISTIC:
            for add_class in utils.HEURISTIC[r["label"]]:
                add(r["video_id"], add_class, r["modality"], meta)
    rows = [
        {
            "video_id": k[0],
            "label": k[1],
            "modality": v["modality"],
            **{m: v[m] for m in utils.META_CLASSES},
        }
        for k, v in acc.items()
    ]
    return pd.DataFrame(rows, columns=pivoted.columns)


# %%
# Save final labels (inhouse + mturk + heuristics) for the selected algorithm.
# This `final` (merge + heuristics) backs the pre-manual _0.1.6 outputs below,
# kept for structural parity with the other generations.
final = get_final_labels(merged, decide_by=DECIDE_BY)
final.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{DECIDE_BY}_inhouse+mturk_formated.csv",
    index=True,
)

# %%
# Pre-manual pivoted labels (v0.1.6), with and without background music.
final_pivoted_premanual = get_pivoted_labels(final)
final_pivoted_premanual.to_csv(
    "supplimentary/data/vggsounder+background-music_0.1.6.csv", index=False
)
final_pivoted_premanual[~final_pivoted_premanual["background_music"]].to_csv(
    "supplimentary/data/vggsounder_0.1.6.csv", index=False
)

# %%
# ----------------------------------------------------------------------------- #
# GEN-4 FINAL CHAIN: Merge -> May 2026 -> Revert Gemini -> Heuristics (last)
# ----------------------------------------------------------------------------- #
# (a) Merged decisions WITHOUT heuristics (heuristics are applied last).
kept_long = get_final_labels(merged, add_heuristics=False, decide_by=DECIDE_BY)
pivoted = get_pivoted_labels(kept_long)
meta_by_video = pivoted.drop_duplicates("video_id").set_index("video_id")[
    utils.META_CLASSES
]

# (b) INTEGRATE MANUAL RE-ANNOTATIONS (may_2026): hard override.
manual = pd.read_csv(
    "supplimentary/manual-annotations/may_2026.csv", keep_default_na=False
)
to_delete_ids = set(manual[manual["label"] == ""]["video_id"])
manual = manual[(manual["label"] != "") & (manual["modality"] != "")]
manual_ids = set(manual["video_id"])

manual_rows = []
for video_id, group in manual.groupby("video_id"):
    meta = _meta_lookup(video_id, meta_by_video)
    for _, r in group.iterrows():
        manual_rows.append(
            {
                "video_id": video_id,
                "label": r["label"],
                "modality": r["modality"],
                **meta,
            }
        )
manual_pivoted = pd.DataFrame(manual_rows, columns=pivoted.columns)

pivoted = pd.concat(
    [
        pivoted[~pivoted["video_id"].isin(manual_ids | to_delete_ids)],
        manual_pivoted,
    ]
)

# (c) Revert ALL Gemini proposals (exists==True), overriding may-2026 deletions.
VERDICTS_PATH = "supplimentary/manual-annotations/deleted_label_verdicts.csv"
verdicts = pd.read_csv(VERDICTS_PATH)
verdicts = verdicts[verdicts["exists"].astype(str).str.lower() == "true"][
    ["video_id", "label", "modality"]
]
n_before = len(pivoted)
pivoted = apply_reverts(pivoted, verdicts, meta_by_video)
print(
    f"Reverted {len(verdicts)} Gemini proposals; "
    f"pivoted rows {n_before} -> {len(pivoted)}"
)

# (d) Heuristics LAST (on merge + may-2026 + reverts).
pivoted = apply_heuristics_pivoted(pivoted)

# (e) Save the final dataset.
pivoted.to_csv("supplimentary/data/vggsounder+background-music.csv", index=False)
pivoted[~pivoted["background_music"]].to_csv(
    "supplimentary/data/vggsounder.csv", index=False
)
