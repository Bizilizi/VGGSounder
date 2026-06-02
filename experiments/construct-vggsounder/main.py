# %%
import utils
import json
import glob
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

pd.set_option("future.no_silent_downcasting", True)


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


IN_JUPYTER = is_notebook()

if not IN_JUPYTER:
    display = print
    plt.show = lambda: None

# CORRECT ORIGINAL LABELS and ADD META LABELS
# %%
# Extract data for original labels and meta labels from goldstandard annotations
# These annotations did not specifically include meta labels, so they have to be
# inferred from the classes
goldstandard = utils.get_gold_standard_df()
display(goldstandard.head())

# %%
# Format inhouse annotation for original labels and meta labels
inhouse = utils.get_inhouse_df()
display(inhouse.head())

# %%
# Get preliminary decisions based on goldstandard and inhouse labels
goldstandard_inhouse = pd.concat([goldstandard, inhouse])

goldstandard_inhouse["is_goldset"] = goldstandard_inhouse["id"].isin(goldstandard["id"])
goldstandard_inhouse_goldset = goldstandard_inhouse[
    goldstandard_inhouse["is_goldset"]
].drop(columns=["is_goldset"])
goldstandard_inhouse_nongoldset = goldstandard_inhouse[
    ~goldstandard_inhouse["is_goldset"]
].drop(columns=["is_goldset"])

# On the goldset, we need to merge decisions from 5 annotations
preliminary_goldset = utils.decide(
    goldstandard_inhouse_goldset, by="dawid-skene", verbose=True
)
preliminary_goldset_OM = preliminary_goldset[
    preliminary_goldset["is_original"] | preliminary_goldset["is_meta"]
]
display(preliminary_goldset_OM.head())
preliminary_goldset_other = preliminary_goldset[
    ~preliminary_goldset["is_original"] & ~preliminary_goldset["is_meta"]
]
display(preliminary_goldset_other.head())

# On everything else, we just have a single annotation
# This is actually just `inhouse` without goldstandard IDs
preliminary_nongoldset_OM = goldstandard_inhouse_nongoldset.drop(columns=["annotator"])
display(preliminary_nongoldset_OM.head())

# %%
# Format results from MTurk run for original and meta labels
mturk_om_run = utils.get_mturk_original_meta_df()
display(mturk_om_run.head())

# %%
# Format results from first MTurk run
# This run included original labels, meta labels, and additional labels
mturk_first_run = utils.get_mturk_df(consider_hits_independent=False)
display(mturk_first_run.head())

# %%
# Merge Mturk results
mturk = pd.concat([mturk_om_run, mturk_first_run])
display(mturk.head())

# %%
# Grade all annotators
# On original and meta labels for goldstandard
# Use agreement (accuracy), since class set is fixed
# Here, we have higher confidence in our preliminary labels, since it's 5 annotators
annotator_scores_goldset_OM = utils.get_annotator_accuracy(
    mturk, preliminary_goldset_OM
)

# On original and meta labels for all but goldstandard
# Use agreement (accuracy), since class set is fixed
# Here, we have lower confidence in our preliminary labels, since it's just 1 annotator
annotator_scores_nongoldset_OM = utils.get_annotator_accuracy(
    mturk, preliminary_nongoldset_OM
)

# On other classes for goldstandard
# Use F1 score, since not all classes might overlap
annotator_scores_goldset_other_acc = utils.get_annotator_accuracy(
    mturk, preliminary_goldset_other
)
annotator_scores_goldset_other_f1 = utils.get_annotator_f1(
    mturk, preliminary_goldset_other
)

fig, axs = plt.subplots(1, 4, figsize=(12, 2), constrained_layout=True)

axs[0].hist(
    list(annotator_scores_goldset_OM.values()),
    bins=50,
    cumulative=True,
    density=True,
    histtype="step",
)
axs[0].set_title("Goldset OM")
axs[1].hist(
    list(annotator_scores_nongoldset_OM.values()),
    bins=50,
    cumulative=True,
    density=True,
    histtype="step",
)
axs[1].set_title("Nongoldset OM")
axs[2].hist(
    list(annotator_scores_goldset_other_acc.values()),
    bins=50,
    cumulative=True,
    density=True,
    histtype="step",
)
axs[2].set_title("Goldset other acc")
axs[3].hist(
    list(annotator_scores_goldset_other_f1.values()),
    bins=50,
    cumulative=True,
    density=True,
    histtype="step",
)
axs[3].set_title("Goldset other f1")
for ax in axs:
    ax.set_xticks(np.arange(0, 1.1, 0.25))
plt.show()

# %%
# Blacklist annotators with aggreement below a threshold
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
print(
    f"Keep in mind that 'annotators' refers to worker-HIT pairs if `consider_hits_independent=True`"
)

# %%
# Merge good annotators with our labels
merged = pd.concat([goldstandard, inhouse, mturk[~mturk["annotator"].isin(blacklist)]])
display(merged.head())


# Make a copy of the merged DataFrame to filter for top labelers
top_labelers_only = merged.copy()
labels = []

# Iterate over each video (grouped by 'id')
for id, group in tqdm(top_labelers_only.groupby("id")):
    # Get unique annotators for this video
    video_labelers = group["annotator"].unique()
    # Get the goldset OM score for each annotator (default to 0 if not found)
    labelers_scores = {
        annotator: annotator_scores_goldset_OM.get(annotator, 0)
        for annotator in video_labelers
    }

    # Select the top 3 annotators with the highest OM scores
    top_labelers = sorted(labelers_scores, key=labelers_scores.get, reverse=True)[:3]

    # Keep only rows for the top 3 annotators for this video
    labels.append(group[group["annotator"].isin(top_labelers)])

# Concatenate all filtered groups back into a single DataFrame
top_labelers_only = pd.concat(labels)

# Calculate statistics about the number of annotators per video after filtering
annotators_per_video = top_labelers_only.copy()
annotators_per_video.drop_duplicates(["id", "annotator"], inplace=True)
annotators_per_video = annotators_per_video.groupby("id").count()["annotator"]

print(
    f"Avg. annotators per video: {annotators_per_video.mean():.2f}, std: {annotators_per_video.std():.2f}, min: {annotators_per_video.min()}, max: {annotators_per_video.max()}"
)

# Update merged to only include the top labelers per video
merged = top_labelers_only.copy()


# %%
def get_final_labels(
    merged, add_heuristics=True, labler_merging=True, decide_by="dawid-skene"
):
    if labler_merging:
        final = utils.decide(merged, by=decide_by, verbose=False)
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
    if add_heuristics:
        avg_nonheuristic_labels = (
            final[~final["is_meta"] & ~final["is_heuristic"]]
            .groupby("id")
            .size()
            .mean()
        )
        print(
            f"Avg. non-meta non-heuristic labels per video: {avg_nonheuristic_labels:.2f}"
        )

    # Get number of videos with each number of labels
    labels_per_video = final[~final["is_meta"]].groupby("id").size()
    labels_per_video = labels_per_video.apply(lambda x: 10 if x >= 10 else x)

    labels_per_video.value_counts().sort_index().plot(kind="bar")
    plt.show()

    return final


# %%
decide_by = "majority"

# Save final labels, e.g. inhouse + mturk + heuristics
final = get_final_labels(merged, decide_by=decide_by)
final.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{decide_by}.csv",
    index=True,
)

# %%
# Save inhouse + heuristics
final_inhouse = get_final_labels(goldstandard_inhouse, labler_merging=False)
final_inhouse.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{decide_by}_inhouse+heuristic.csv",
    index=True,
)
# %%
# Save inhouse + mturk
final_inhouse_mturk = get_final_labels(
    merged, add_heuristics=False, decide_by=decide_by
)
final_inhouse_mturk.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{decide_by}_inhouse+mturk.csv",
    index=True,
)


# %%
# get pivoted labels function
def get_pivoted_labels(final):
    # Save labels in pivoted format
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
# Save pivoted labels
final_pivoted = get_pivoted_labels(final)
final_pivoted.to_csv(
    f"supplimentary/data/vggsounder+background-music_0.1.6.csv",
    index=False,
)

vggsounder_v016 = final_pivoted[~final_pivoted["background_music"]]
vggsounder_v016.to_csv(
    f"supplimentary/data/vggsounder_0.1.6.csv",
    index=False,
)

# %%
# Save inhouse + heuristics pivoted labels
final_inhouse_pivoted = get_pivoted_labels(final_inhouse)
final_inhouse_pivoted.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{decide_by}_inhouse+heuristic_formated.csv",
    index=False,
)

# %%
# Save inhouse + mturk pivoted labels
final_inhouse_mturk_pivoted = get_pivoted_labels(final_inhouse_mturk)
final_inhouse_mturk_pivoted.to_csv(
    f"supplimentary/intermediate-tables/{threshold:.1f}_{decide_by}_inhouse+mturk_formated.csv",
    index=False,
)

# %%
# INTEGRATE MANUAL RE-ANNOTATIONS (may_2026) -> VGGSounder v0.1.6
# may_2026.csv has only label + modality; meta labels are preserved per-video
# from the unchanged pipeline output `final_pivoted`.
manual = pd.read_csv(
    "supplimentary/manual-annotations/may_2026.csv", keep_default_na=False
)

# Unannotated samples -> delete set (derived inline from may_2026.csv)
to_delete_ids = set(manual[manual["label"] == ""]["video_id"])

# Real replacement annotations
manual = manual[(manual["label"] != "") & (manual["modality"] != "")]
manual_ids = set(manual["video_id"])

meta_by_video = final_pivoted.drop_duplicates("video_id").set_index("video_id")[
    utils.META_CLASSES
]

manual_rows = []
for video_id, group in manual.groupby("video_id"):
    meta = (
        meta_by_video.loc[video_id].to_dict()
        if video_id in meta_by_video.index
        else {m: False for m in utils.META_CLASSES}
    )
    for _, r in group.iterrows():
        manual_rows.append(
            {
                "video_id": video_id,
                "label": r["label"],
                "modality": r["modality"],
                **meta,
            }
        )
manual_pivoted = pd.DataFrame(manual_rows, columns=final_pivoted.columns)

final_pivoted = pd.concat(
    [
        final_pivoted[~final_pivoted["video_id"].isin(manual_ids | to_delete_ids)],
        manual_pivoted,
    ]
)

final_pivoted.to_csv(
    "supplimentary/data/vggsounder+background-music.csv", index=False
)
vggsounder = final_pivoted[~final_pivoted["background_music"]]
vggsounder.to_csv("supplimentary/data/vggsounder.csv", index=False)
