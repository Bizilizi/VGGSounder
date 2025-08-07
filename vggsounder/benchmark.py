from collections import defaultdict
import glob
import os
import pickle as pk
from typing import List, Literal
import uuid
from pathlib import Path

# Data manipulation and numerical computing
import numpy as np
import pandas as pd

# Machine learning metrics and preprocessing
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

from vggsounder.labels import VGGSounder

# Supported Modalities for VGGSounder evaluation
# Order matters for consistent processing and visualization
MODALITIES = [
    "a",  # Audio: includes AV samples with audio component (A + AV)
    "v",  # Visual: includes AV samples with visual component (V + AV)
    "av",  # Audio-Visual: samples with both modalities (AV only)
    "a only",  # Audio-only: pure audio samples (excludes AV samples)
    "v only",  # Visual-only: pure visual samples (excludes AV samples)
]

# List of evaluation metrics supported by the benchmark
# Includes top-k metrics, macro/micro averages, AUC metrics, and modality confusion
METRICS = [
    # Hit rate metrics (top-k accuracy)
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "hit_rate@10",
    "hit_rate",
    # F1 score metrics (top-k and overall)
    "f1@1",
    "f1@3",
    "f1@5",
    "f1@10",
    "f1",
    # Macro F1 score metrics
    "f1_macro@1",
    "f1_macro@3",
    "f1_macro@5",
    "f1_macro@10",
    "f1_macro",
    # Jaccard similarity metrics (top-k and overall)
    "jaccard@1",
    "jaccard@3",
    "jaccard@5",
    "jaccard@10",
    "jaccard",
    # Precision metrics (top-k and overall)
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    "precision",
    # Recall metrics (top-k and overall)
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "recall",
    # Accuracy metrics (top-k and overall)
    "accuracy@1",
    "accuracy@3",
    "accuracy@5",
    "accuracy@10",
    "accuracy",
    # Area under curve metrics
    "auc_roc",  # ROC-AUC
    "auc_pr",  # Precision-Recall AUC
    # Prediction statistics (not currently implemented)
    "avg_n_preds",  # Average number of predictions per sample
    "med_n_preds",  # Median number of predictions per sample
    # Modality confusion metric
    "mu",  # Modality understanding confusion rate (computed separately)
]

# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================


def get_gt_for_modality(
    vggsounder: VGGSounder,
    modality: Literal["av", "a", "v", "a only", "v only"],
):
    """Extract ground truth labels for a specific modality subset.

    Args:
        vggsounder (VGGSounder): VGGSounder dataset instance
        modality (Literal): Target modality to extract:
            - 'av': Audio-visual samples only
            - 'a': All samples with audio component (A + AV)
            - 'v': All samples with visual component (V + AV)
            - 'a only': Audio-only samples (A)
            - 'v only': Visual-only samples (V)

    Returns:
        pd.DataFrame: Processed dataframe with columns ['id', 'labels']
            where 'labels' contains list of labels per sample ID

    Note:
        Uses VGGSounder.set_modality() to filter the dataset for the specified modality.
    """
    vggsounder.set_modality(modality)

    df_rows = []
    for video_data in vggsounder:
        df_rows.append(
            {
                "id": video_data.video_id,
                "labels": video_data.labels,
            }
        )

    df = pd.DataFrame(df_rows)
    return df


def load_test_predictions(
    model: str,
    modality: Literal["av", "a", "v", "a only", "v only"],
) -> pd.DataFrame:
    """Load model predictions for a specific modality from pickle file.

    Args:
        model (str): Path to the model's pickle file containing predictions
        modality (Literal): Target modality for predictions

    Returns:
        pd.DataFrame: Predictions dataframe with columns ['id', 'labels', 'logits', 'modality']
            where 'labels' contains list of predicted labels and 'logits' contains prediction scores

    Note:
        - For 'a only' and 'v only', maps to 'a' and 'v' respectively in the pickle file
        - Loads predictions and logits from nested dictionary structure
        - Video IDs have '.mp4' extension stripped for consistency
        - Returns both discrete predictions and continuous logits when available
    """
    # Map modality aliases to actual modality names
    if modality == "a only":
        modality = "a"
    elif modality == "v only":
        modality = "v"

    predictions = pk.load(open(model, "rb"))
    df_rows = []
    for id, predictions_dict in predictions.items():
        if "logits" in predictions_dict:
            logits = predictions_dict["logits"][modality]
        else:
            logits = None

        if "predictions" in predictions_dict:
            predictions = predictions_dict["predictions"][modality]
        else:
            predictions = None

        df_rows.append(
            {
                "id": id.replace(".mp4", ""),
                "labels": predictions,
                "logits": logits,
                "modality": modality,
            }
        )

    df = pd.DataFrame(df_rows)
    return df


# =============================================================================
# METRICS COMPUTATION FUNCTIONS
# =============================================================================


def compare_labels(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, classes: list[str]
) -> dict[str, float]:
    """Compare ground truth and predicted labels using multiple metrics.

    Args:
        gt_df (pd.DataFrame): Ground truth with columns ['id', 'labels']
        pred_df (pd.DataFrame): Predictions with columns ['id', 'labels']
        classes (list[str]): List of all possible class names

    Returns:
        dict[str, float]: Dictionary containing computed metrics:
            - f1: Micro-averaged F1 score
            - f1_macro: Macro-averaged F1 score
            - accuracy: Exact match accuracy (all labels must match)
            - jaccard: Micro-averaged Jaccard similarity (IoU)
            - precision: Micro-averaged precision
            - recall: Micro-averaged recall
            - hit_rate: Proportion of samples with at least one correct prediction

    Note:
        - Uses MultiLabelBinarizer to convert label lists to binary matrices
        - Missing predictions are treated as empty label lists
        - All metrics handle multi-label classification scenarios
        - Zero division cases are handled gracefully
    """
    # Merge ground truth and predictions on sample ID
    merged_df = pd.merge(gt_df, pred_df, on="id", how="left", suffixes=("_gt", "_pred"))

    # Handle missing predictions by replacing NaN with empty lists
    def _fill_na_empty_list(x):
        """Convert NaN values to empty lists for consistent processing."""
        if isinstance(x, list):
            return x
        return []

    merged_df["labels_gt"] = merged_df["labels_gt"].apply(_fill_na_empty_list)
    merged_df["labels_pred"] = merged_df["labels_pred"].apply(_fill_na_empty_list)

    # Convert label lists to binary matrices
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])  # Initialize with empty list to include all classes
    y_true = mlb.transform(merged_df["labels_gt"])
    y_pred = mlb.transform(merged_df["labels_pred"])

    # Compute classification metrics
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)  # Exact match accuracy
    jaccard = jaccard_score(y_true, y_pred, average="micro")
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)

    # Compute hit rate (at least one correct prediction per sample)
    hits = np.any(y_true & y_pred, axis=1)
    hit_rate = np.mean(hits)

    return {
        "f1": f1,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
    }


def compare_labels_logits(
    gt_df: pd.DataFrame,
    logits_df: pd.DataFrame,
    classes: list[str],
    ks: List[int] = [1, 3, 5, 10],
) -> dict[str, float]:
    """Compare ground truth labels with model logits using top-k metrics and AUC.

    Args:
        gt_df (pd.DataFrame): Ground truth with columns ['id', 'labels']
        logits_df (pd.DataFrame): Model logits with columns ['id', 'logits']
        classes (list[str]): List of all possible class names
        ks (List[int]): List of k values for top-k evaluation (default: [1, 3, 5, 10])

    Returns:
        dict[str, float]: Dictionary containing computed metrics:
            - Top-k metrics: f1@k, f1_macro@k, accuracy@k, jaccard@k,
              precision@k, recall@k, hit_rate@k for each k in ks
            - AUC metrics: auc_roc (ROC-AUC), auc_pr (Precision-Recall AUC)

    Note:
        - Uses logits to rank predictions by sorting in ascending order (highest k values)
        - AUC metrics are computed per class then macro-averaged across valid classes
        - Skips classes with insufficient positive examples for AUC computation
        - Samples without logits are excluded from evaluation
        - Top-k predictions are created by setting top k logit values to 1, others to 0
    """
    # Merge ground truth and logits on sample ID
    merged_df = pd.merge(gt_df, logits_df, on="id", how="left")
    merged_df = merged_df.dropna()  # Remove samples without logits

    # Convert labels to binary matrix
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])
    y_true = mlb.transform(merged_df["labels"])

    # Convert logits list to numpy array
    logits_array = np.array(merged_df["logits"].tolist())

    # Calculate top-k metrics for each specified k
    results = {}
    for k in ks:
        # Get top-k predictions by sorting logits (ascending order, take last k)
        top_k = np.argsort(logits_array, axis=1)[:, -k:]  # Indices of highest k logits
        y_pred = np.zeros_like(logits_array, dtype=int)
        np.put_along_axis(
            y_pred, top_k, 1, axis=1
        )  # Binary matrix: 1 for top-k, 0 elsewhere

        # Compute metrics for this k
        results[f"f1@{k}"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        results[f"f1_macro@{k}"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        results[f"accuracy@{k}"] = accuracy_score(y_true, y_pred)
        results[f"jaccard@{k}"] = jaccard_score(y_true, y_pred, average="micro")
        results[f"precision@{k}"] = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        results[f"recall@{k}"] = recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )

        # Compute hit rate (at least one correct in top-k)
        hits = np.any(y_true & y_pred, axis=1)
        results[f"hit_rate@{k}"] = np.mean(hits)

    # Calculate Area Under Curve metrics per class
    auc_roc_scores = []
    auc_pr_scores = []

    for i in range(len(classes)):
        # Only calculate AUC for classes that exist in ground truth
        # (AUC requires at least one positive example)
        if np.sum(y_true[:, i]) > 0:
            try:
                # ROC-AUC: Area under Receiver Operating Characteristic curve
                auc_roc = roc_auc_score(y_true[:, i], logits_array[:, i])
                # PR-AUC: Area under Precision-Recall curve
                auc_pr = average_precision_score(y_true[:, i], logits_array[:, i])
                auc_roc_scores.append(auc_roc)
                auc_pr_scores.append(auc_pr)
            except ValueError:
                # Skip classes with degenerate cases (all 0s or all 1s)
                pass

    # Average AUC scores across valid classes
    if auc_roc_scores:
        results["auc_roc"] = np.mean(auc_roc_scores)
    else:
        results["auc_roc"] = 0.0

    if auc_pr_scores:
        results["auc_pr"] = np.mean(auc_pr_scores)
    else:
        results["auc_pr"] = 0.0

    return results


# =============================================================================
# MODALITY CONFUSION ANALYSIS FUNCTIONS
# =============================================================================


def compute_modality_confusion(
    gt_df: pd.DataFrame,
    pred_df_av: pd.DataFrame,
    pred_df_a: pd.DataFrame,
    pred_df_v: pd.DataFrame,
    classes: list[str],
) -> dict[str, float]:
    """Compute modality confusion rates for multi-modal model evaluation.

    Measures how often single-modality predictions succeed when the combined
    audio-visual prediction fails, indicating potential modality confusion.

    Args:
        gt_df (pd.DataFrame): Ground truth labels with columns ['id', 'labels']
        pred_df_av (pd.DataFrame): Audio-visual predictions with columns ['id', 'labels']
        pred_df_a (pd.DataFrame): Audio-only predictions with columns ['id', 'labels']
        pred_df_v (pd.DataFrame): Visual-only predictions with columns ['id', 'labels']
        classes (list[str]): List of all possible class names

    Returns:
        dict[str, float]: Confusion rates (as fractions, not percentages):
            - 'a': Audio confusion rate (audio hits when AV fails)
            - 'v': Visual confusion rate (visual hits when AV fails)
            - 'av': Combined confusion rate (both A and V hit when AV fails)

    Note:
        - Higher confusion rates suggest the model struggles to properly
          integrate multi-modal information compared to single modalities
        - Uses hit rate computation (at least one correct prediction per sample)
        - Missing predictions are treated as empty label lists
    """
    # Merge all prediction dataframes with ground truth
    merged_df = pd.merge(
        gt_df, pred_df_av, on="id", how="left", suffixes=("_gt", "_av")
    )
    merged_df = pd.merge(merged_df, pred_df_a, on="id", how="left")
    merged_df = merged_df.rename(columns={"labels": "labels_a"})
    merged_df = pd.merge(merged_df, pred_df_v, on="id", how="left")
    merged_df = merged_df.rename(columns={"labels": "labels_v"})

    # Handle missing predictions
    def _fill_na_empty_list(x):
        """Convert NaN values to empty lists for consistent processing."""
        if isinstance(x, list):
            return x
        return []

    for col in ["labels_gt", "labels_av", "labels_a", "labels_v"]:
        merged_df[col] = merged_df[col].apply(_fill_na_empty_list)

    # Convert to binary matrices for hit computation
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])
    y_true = mlb.transform(merged_df["labels_gt"])
    y_pred_av = mlb.transform(merged_df["labels_av"])
    y_pred_a = mlb.transform(merged_df["labels_a"])
    y_pred_v = mlb.transform(merged_df["labels_v"])

    # Compute hit indicators (at least one correct prediction per sample)
    hits_av = np.any(y_true & y_pred_av, axis=1)  # AV hits
    hits_a = np.any(y_true & y_pred_a, axis=1)  # Audio hits
    hits_v = np.any(y_true & y_pred_v, axis=1)  # Visual hits

    # Compute confusion rates (single modality succeeds when AV fails)
    confused_a = np.mean(hits_a & ~hits_av)  # Audio confusion
    confused_v = np.mean(hits_v & ~hits_av)  # Visual confusion
    confused_av = np.mean(hits_a & hits_v & ~hits_av)  # Both succeed, AV fails

    return {
        "a": confused_a,
        "v": confused_v,
        "av": confused_av,
    }


def compute_modality_confusion_logits(
    gt_df: pd.DataFrame,
    logit_df_av: pd.DataFrame,
    logit_df_a: pd.DataFrame,
    logit_df_v: pd.DataFrame,
    classes: list[str],
) -> dict[str, float]:
    """Compute modality confusion rates using model logits with top-1 predictions.

    Similar to compute_modality_confusion but uses logits to determine
    predictions by taking the top-1 (highest scoring) class for each modality.

    Args:
        gt_df (pd.DataFrame): Ground truth labels with columns ['id', 'labels']
        logit_df_av (pd.DataFrame): Audio-visual logits with columns ['id', 'logits']
        logit_df_a (pd.DataFrame): Audio-only logits with columns ['id', 'logits']
        logit_df_v (pd.DataFrame): Visual-only logits with columns ['id', 'logits']
        classes (list[str]): List of all possible class names

    Returns:
        dict[str, float]: Confusion rates based on top-1 predictions (as fractions):
            - 'a': Audio confusion rate (audio top-1 hits when AV top-1 fails)
            - 'v': Visual confusion rate (visual top-1 hits when AV top-1 fails)
            - 'av': Combined confusion rate (both A and V top-1 hit when AV top-1 fails)

    Note:
        - Uses top-1 predictions only (highest logit value) for each modality
        - Samples without logits for any modality are excluded from analysis
        - Creates binary prediction matrices with only the highest-scoring class set to 1
    """
    # Merge all logit dataframes with ground truth
    merged_df = pd.merge(gt_df, logit_df_av, on="id", how="left")
    merged_df = merged_df.rename(columns={"logits": "logits_av"})
    merged_df = pd.merge(merged_df, logit_df_a, on="id", how="left")
    merged_df = merged_df.rename(columns={"logits": "logits_a"})
    merged_df = pd.merge(merged_df, logit_df_v, on="id", how="left")
    merged_df = merged_df.rename(columns={"logits": "logits_v"})
    merged_df = merged_df.dropna()  # Remove samples missing any logits

    # Convert ground truth labels to binary matrix
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([])
    y_true = mlb.transform(merged_df["labels"])

    # Convert logits to numpy arrays
    logits_array_av = np.array(merged_df["logits_av"].tolist())
    logits_array_a = np.array(merged_df["logits_a"].tolist())
    logits_array_v = np.array(merged_df["logits_v"].tolist())

    # Get top-1 predictions (highest logit) for each modality
    top_k_av = np.argsort(logits_array_av, axis=1)[:, -1:]  # Top-1 indices
    top_k_a = np.argsort(logits_array_a, axis=1)[:, -1:]
    top_k_v = np.argsort(logits_array_v, axis=1)[:, -1:]

    # Create binary prediction matrices
    y_pred_av = np.zeros_like(logits_array_av, dtype=int)
    y_pred_a = np.zeros_like(logits_array_a, dtype=int)
    y_pred_v = np.zeros_like(logits_array_v, dtype=int)

    # Set top-1 predictions to 1
    np.put_along_axis(y_pred_av, top_k_av, 1, axis=1)
    np.put_along_axis(y_pred_a, top_k_a, 1, axis=1)
    np.put_along_axis(y_pred_v, top_k_v, 1, axis=1)

    # Compute hit indicators
    hits_av = np.any(y_true & y_pred_av, axis=1)
    hits_a = np.any(y_true & y_pred_a, axis=1)
    hits_v = np.any(y_true & y_pred_v, axis=1)

    # Compute confusion rates
    confused_a = np.mean(hits_a & ~hits_av)
    confused_v = np.mean(hits_v & ~hits_av)
    confused_av = np.mean(hits_a & hits_v & ~hits_av)

    return {
        "a": confused_a,
        "v": confused_v,
        "av": confused_av,
    }


# =============================================================================
# BENCHMARK COMPUTATION FUNCTIONS
# =============================================================================


def compute_all_results(
    models_path: str,
    out: str | None = None,
    *,
    dataset_path: str | None = None,
    subset_ids: list[str] | None = None,
    modalities: list[str] = ["av", "a", "v", "a only", "v only"],
    verbose: bool = True,
    models_filter: list[str] | None = None,
):
    """Compute benchmark results for all models and modalities.

    This is the main benchmarking function that evaluates models across all
    specified modalities using multiple metrics including logits-based evaluation.

    Args:
        out (str): Output path for pickled results. If None, results are not saved.
        models_path (str): Path to directory containing model pickle files
        dataset_path (str | None): Path to VGGSounder dataset (if None, uses default)
        subset_ids (list[str] | None): List of video IDs to evaluate on
            (if None, uses all samples)
        modalities (list[str]): List of modalities to evaluate
        models_filter (list[str] | None): List of model names to evaluate
            (if None, uses all models)

    Returns:
        dict: Nested dictionary with benchmark results
            Structure: {model_name: {modality: {metric: value}}}

    Note:
        - Automatically detects whether models have logits and uses appropriate evaluation
        - Models with logits use compare_labels_logits for detailed top-k metrics
        - Models without logits use compare_labels for basic metrics
        - Modality confusion analysis is performed for all models with av, a, v predictions
        - Results include metrics like F1, accuracy, hit rate, AUC, and modality confusion
        - Progress is tracked with tqdm progress bars
        - All .pkl files in models_path are automatically processed
    """
    # Load and prepare ground truth data
    vggsounder = VGGSounder(dataset_path)
    vggsounder_df = pd.read_csv(
        vggsounder.csv_path
    )  # covers the case where dataset_path is None

    if subset_ids is not None:
        vggsounder_df = vggsounder_df[vggsounder_df["video_id"].isin(subset_ids)]
    vggsounder_df.rename(columns={"video_id": "id"}, inplace=True)

    # Load class names and prepare ground truth for each modality
    classes = vggsounder.get_all_labels()
    gts = {
        modality: get_gt_for_modality(vggsounder, modality) for modality in modalities
    }

    all_results = defaultdict(dict)

    # Evaluate foundation models (discrete predictions)
    models_files = glob.glob(os.path.join(models_path, "*.pkl"))
    model_names = [os.path.splitext(os.path.basename(model_file))[0] for model_file in models_files]
    
    if models_filter is not None:
        models_files = [
            model_file
            for name, model_file in zip(model_names, models_files)
            if name in models_filter
        ]
        model_names = [name for name in model_names if name in models_filter]
        
    tqdm_iterator = tqdm(zip(model_names, models_files), total=len(models_files), disable=not verbose, leave=False)

    for model, model_file in tqdm_iterator:

        # Set description for tqdm
        tqdm_iterator.set_description(model)

        # Load predictions for all modalities
        preds = {
            modality: load_test_predictions(model_file, modality)
            for modality in modalities
        }

        # Compute metrics for each modality
        for modality in modalities:
            gt = gts[modality]
            modality_pred = preds[modality]
            contains_logits = not modality_pred["logits"].isna().any()

            # Skip modalities without predictions (not all models support all modalities)
            if len(modality_pred) == 0:
                continue

            if contains_logits:
                modality_pred.drop(columns=["labels"], inplace=True)
                results = compare_labels_logits(gt, modality_pred, classes)
            else:
                modality_pred.drop(columns=["logits"], inplace=True)
                results = compare_labels(gt, modality_pred, classes)

            all_results[model][modality] = results

        # Compute modality confusion analysis
        if contains_logits:
            mus = compute_modality_confusion_logits(
                gts["av"], preds["av"], preds["a"], preds["v"], classes
            )
        else:
            mus = compute_modality_confusion(
                gts["av"], preds["av"], preds["a"], preds["v"], classes
            )
        for modality in ["av", "a", "v"]:
            all_results[model][modality]["mu"] = mus[modality]

    # Save results to pickle file
    if out is not None:
        pk.dump(all_results, open(out, "wb"))

    return all_results


# =============================================================================
# RESULT PROCESSING AND FORMATTING FUNCTIONS
# =============================================================================


def select_metrics(df, metrics, display_names=None):
    """Select specific metrics from a multi-level indexed dataframe.

    Args:
        df (pd.DataFrame): Multi-level indexed dataframe with metrics as columns
        metrics (list): List of metric names or (metric, modalities) tuples to select
        display_names (dict, optional): Mapping from model names to display names

    Returns:
        pd.DataFrame: Filtered dataframe with selected metrics and renamed models

    Note:
        - Handles both single metric names and (metric, modalities) tuples
        - For tuples, extracts metric and modality combinations using pandas IndexSlice
        - Supports renaming models using display_names mapping for better visualization
        - Works with pandas MultiIndex columns created by get_metric_dataframe
        - Preserves the hierarchical structure of the original dataframe
    """
    idx = pd.IndexSlice
    parts = []

    # Extract requested metrics
    for metric in metrics:
        if isinstance(metric, tuple):
            # Handle (metric, modalities) tuple format
            m, mods = metric
            parts.append(df.loc[:, idx[m, mods, :]])
        else:
            # Handle single metric name
            if metric in df.columns.get_level_values(0):
                parts.append(df.loc[:, idx[metric, :, :]])
            elif metric in df.columns:
                parts.append(df[[metric]])

    # Combine selected metrics
    df = pd.concat(parts, axis=1)

    # Apply display name mapping if provided
    if display_names is not None:
        df = df.loc[list(display_names.keys())]
        df.index = [display_names.get(i, i) for i in df.index]

    return df


def get_metric_dataframe(all_results, modalities=None):
    """Convert nested results dictionary to a structured pandas DataFrame.

    Transforms the nested results structure {model: {modality: {metric: value}}}
    into a multi-level indexed DataFrame for easier analysis and visualization.

    Args:
        all_results (dict): Nested dictionary with benchmark results
            Structure: {model_name: {modality: {metric_name: value}}}
        modalities (list, optional): List of modalities to include
            (defaults to MODALITIES constant)

    Returns:
        pd.DataFrame: Multi-level indexed DataFrame with:
            - Rows: Model names
            - Columns: Multi-level index with (metric, modality) pairs
            - Values: Metric scores scaled to percentage (×100)

    Note:
        - Missing metrics are filled with None values
        - Falls back to "metric@1" format for metrics without explicit values
        - Values are automatically scaled to percentages (×100) for display
        - Creates a pivoted table structure from the nested dictionary
        - Handles the METRICS and MODALITIES constants for coverage
    """
    # Restructure nested dictionary for easier DataFrame creation
    all_results_inverted = {
        model: {
            metric: {
                modality: all_results[model][modality].get(
                    metric, all_results[model][modality].get(f"{metric}@1", None)
                )
                for modality in (modalities or MODALITIES)
            }
            for metric in METRICS
        }
        for model in all_results
    }

    # Create lists for multi-index construction
    index_tuples = []
    values = []

    # Flatten the nested structure
    for model in all_results_inverted:
        for metric in all_results_inverted[model]:
            for modality in all_results_inverted[model][metric]:
                # Create tuple for multi-index: (model, metric, modality)
                index_tuples.append((model, metric, modality))

                # Scale values to percentage and handle None values
                if all_results_inverted[model][metric][modality] is not None:
                    values.append(all_results_inverted[model][metric][modality] * 100)
                else:
                    values.append(None)

    # Create multi-index DataFrame
    index = pd.MultiIndex.from_tuples(
        index_tuples, names=["model", "metric", "modality"]
    )
    df = pd.Series(values, index=index)

    # Pivot to get metrics and modalities as columns
    df_table = df.unstack(level=["metric", "modality"])

    return df_table


def benchmark(
    models_path: str,
    display_names: dict[str, str],
    metrics: list[tuple[str, list[str]]],
    verbose: bool = True,
):
    """
    Run the VGGSounder benchmark and return a formatted metrics table.

    This function computes evaluation metrics for a set of models and modalities,
    then selects and formats the results for display or further analysis.

    Args:
        models_path (str):
            Path to the directory containing the model results.
            The directory should contain .pkl files with model predictions.
            The file names should be the model names.

        display_names (dict[str, str]):
            A mapping from internal model names (as used in the results) to
            human-friendly display names for presentation in tables or plots.

            Example:
                display_names = {
                    "cav-mae": "CAV-MAE",
                    "deepavfusion": "DeepAVFusion",
                    "equiav": "Equi-AV",
                    "avsiam": "AV-Siam",
                    "gemini-1.5-flash": "Gemini 1.5 Flash",
                    "gemini-1.5-pro": "Gemini 1.5 Pro",
                    "gemini-2.0-flash": "Gemini 2.0 Flash",
                    "video-llama-2-av": "VideoLLaMA 2",
                    "unified-io-2": "Unified-IO 2",
                    "pandagpt": "PandaGPT",
                    "ola": "OLA",
                }

        metrics (list[tuple[str, list[str]]]):
            A list of tuples specifying which metrics to extract and for which modalities.
            Each tuple is of the form (metric_name, [modalities]), where metric_name is a string
            (e.g., "accuracy", "f1", "hit_rate", "mu") and [modalities] is a list of modality names
            (e.g., ["a", "v", "av", "a only", "v only"]).

            Example:
                metrics = [
                    ("accuracy", ["a", "v", "av"]),
                    ("f1", ["a", "v", "av", "a only", "v only"]),
                    ("hit_rate", ["a", "v", "av"]),
                    ("mu", ["a", "v", "av"])
                ]

            This input tells the function to extract the "accuracy" metric for modalities "a", "v", and "av";
            the "f1" metric for all five modalities; the "hit_rate" for "a", "v", and "av"; and the "mu"
            (modality confusion) metric for "a", "v", and "av".

        verbose (bool, optional):
            If True, prints progress bars and status messages during computation.
            Default is True.

    Returns:
        pd.DataFrame:
            A table (DataFrame) where rows correspond to models (using display names if provided),
            and columns correspond to the selected metrics and modalities. The values are the
            computed metric scores (typically as percentages).

    What this function does:
        1. Computes all benchmark results for available models and modalities using `compute_all_results`.
        2. Converts the nested results into a DataFrame with `get_metric_dataframe`.
        3. Selects and formats the requested metrics and modalities using `select_metrics`, applying
           the provided display names for models.
        4. Returns the resulting table for display or further analysis.

    Example usage:
        >>> display_names = {
        ...     "cav-mae": "CAV-MAE",
        ...     "deepavfusion": "DeepAVFusion",
        ...     "equiav": "Equi-AV",
        ... }
        >>> metrics = [
        ...     ("accuracy", ["a", "v", "av"]),
        ...     ("f1", ["a", "v", "av", "a only", "v only"]),
        ...     ("hit_rate", ["a", "v", "av"]),
        ...     ("mu", ["a", "v", "av"])
        ... ]
        >>> table = benchmark(display_names, metrics)
        >>> print(table)

    """
    results = compute_all_results(
        models_path=models_path,
        models_filter=set(display_names.keys()) if display_names is not None else None,
        verbose=verbose,
    )

    df = get_metric_dataframe(results)
    table = select_metrics(df, metrics, display_names=display_names)

    return table
