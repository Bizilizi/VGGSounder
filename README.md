<h1 align="center"><a href="https://vggsounder.github.io/static/workshop_paper.pdf">
VGGSounder: Audio-Visual Evaluations for Foundation Models</a></h1>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏</h2>


<h5 align="center">

<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.13106)  -->
[![Project page](https://img.shields.io/badge/Project_page-https-blue)](https://vggsounder.github.io) 
<br>

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/LICENSE) 
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2FBizilizi%2Fvggsounder&label=HITs&icon=fire&color=%23198754)
[![GitHub issues](https://img.shields.io/github/issues/Bizilizi/vggsounder?color=critical&label=Issues)](https://github.com/Bizilizi/vggsounder/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/Bizilizi/vggsounder?color=success&label=Issues)](https://github.com/Bizilizi/vggsounder/issues?q=is%3Aissue+is%3Aclosed)
</h5>

## 📰 News

* **[11.06.2025]**  📃 Released technical report of VGGSounder. Contains detailed discussion on how we built the first multimodal benchmark for video tagging with complete per-modality annotations for every class.


## 🌟 Introduction
**VGGSounder** is a re-annotated benchmark built upon the [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), designed to rigorously evaluate audio-visual foundation models and understand how they utilize modalities. VGGSounder introduces:

- 🔍 Per-label modality tags (audible / visible / both) for all classes in the sample
- 🎵 Meta labels for background music, voice-over, and static images
- 📊 Multiple classes per one sample


## 🚀 Installation

The VGGSounder dataset is now available as a Python package! Install it via pip:

```bash
pip install vggsounder
```

Or install from source using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/bizilizi/vggsounder.git
cd vggsounder
uv build
pip install dist/vggsounder-*.whl
```

## 🐍 Python Package Usage

### Quick Start

```python
import vggsounder

# Load the dataset
labels = vggsounder.VGGSounder()

# Access video data by ID
video_data = labels["--U7joUcTCo_000000"]
print(video_data.labels)        # List of labels for this video
print(video_data.meta_labels)   # Metadata (background_music, static_image, voice_over)
print(video_data.modalities)    # Modality for each label (A, V, AV)

# Get dataset statistics
stats = labels.stats()
print(f"Total videos: {stats['total_videos']}")
print(f"Unique labels: {stats['unique_labels']}")

# Search functionality
piano_videos = labels.get_videos_with_labels("playing piano")
voice_over_videos = labels.get_videos_with_meta(voice_over=True)
```

### Advanced Usage

```python
# Dict-like interface
print(len(labels))                    # Number of videos
print("video_id" in labels)           # Check if video exists
for video_id in labels:               # Iterate over video IDs
    video_data = labels[video_id]

# Get all unique labels
all_labels = labels.get_all_labels()

# Complex queries
static_speech_videos = labels.get_videos_with_meta(
    static_image=True, voice_over=True
)
```

## 🏷️ Label Format

VGGSounder annotations are stored in a CSV file located at `data/vggsounder.csv`. Each row corresponds to a single label for a specific video sample. The dataset supports **multi-label**, **multi-modal** classification with additional **meta-information** for robust evaluation.

### Columns

- **`video_id`**: Unique identifier for a 10-second video clip.
- **`label`**: Human-readable label representing a sound or visual category (e.g. `male singing`, `playing timpani`).
- **`modality`**: The modality in which the label is perceivable:
  - `A` = Audible
  - `V` = Visible
  - `AV` = Both audible and visible
- **`background_music`**: `True` if the video contains background music.
- **`static_image`**: `True` if the video consists of a static image.
- **`voice_over`**: `True` if the video contains voice-over narration.

### Example

| video_id           | label             | modality | background_music | static_image | voice_over |
|--------------------|------------------|----------|------------------|--------------|------------|
| `---g-f_I2yQ_000001` | `male singing`     | A        | True             | False        | False      |
| `---g-f_I2yQ_000001` | `people crowd`     | AV       | True             | False        | False      |
| `---g-f_I2yQ_000001` | `playing timpani`  | A        | True             | False        | False      |

## 🧪 Benchmark Evaluation

VGGSounder provides a comprehensive benchmarking system to evaluate audio-visual foundation models across multiple modalities and metrics. The benchmark supports both discrete predictions and continuous logits-based evaluation.

### Supported Modalities

- **`a`**: Audio - includes samples with audio component (A + AV)
- **`v`**: Visual - includes samples with visual component (V + AV)
- **`av`**: Audio-Visual - samples with both modalities (AV only)
- **`a only`**: Audio-only - pure audio samples (excludes AV samples)
- **`v only`**: Visual-only - pure visual samples (excludes AV samples)

### Available Metrics

The benchmark computes a comprehensive set of metrics:
- **Top-k metrics**: `hit_rate@k`, `f1@k`, `accuracy@k`, `precision@k`, `recall@k`, `jaccard@k` (for k=1,3,5,10)
- **Aggregate metrics**: `f1`, `f1_macro`, `accuracy`, `precision`, `recall`, `jaccard`, `hit_rate`
- **AUC metrics**: `auc_roc`, `auc_pr` (ROC-AUC and Precision-Recall AUC)
- **Modality confusion**: `mu` (measures when single modalities succeed where multimodal fails)

### Model Results Format

Model predictions should be saved as pickle files with the following structure:

```python
{
    "video_id": {
        "predictions": {  # Optional: discrete predictions
            "a": ["label1", "label2", ...],     # Audio predictions
            "v": ["label1", "label3", ...],     # Visual predictions
            "av": ["label1", "label2", ...]     # Audio-visual predictions
        },
        "logits": {      # Optional: continuous scores
            "a": [0.1, 0.8, 0.3, ...],         # Audio logits (310 classes)
            "v": [0.2, 0.1, 0.9, ...],         # Visual logits (310 classes)  
            "av": [0.4, 0.6, 0.2, ...]         # Audio-visual logits (310 classes)
        }
    },
    # ... more video_ids
}
```

**Note**: Either `predictions` or `logits` (or both) should be provided. Logits enable more detailed top-k and AUC analysis.

### Running the Benchmark

#### Quick Start

```python
from vggsounder.benchmark import benchmark

# Define model display names
display_names = {
    "cav-mae": "CAV-MAE",
    "deepavfusion": "DeepAVFusion", 
    "equiav": "Equi-AV",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-pro": "Gemini 1.5 Pro"
}

# Specify metrics and modalities to evaluate
metrics = [
    ("accuracy", ["a", "v", "av"]),
    ("f1", ["a", "v", "av", "a only", "v only"]), 
    ("hit_rate", ["a", "v", "av"]),
    ("mu", ["a", "v", "av"])  # Modality confusion
]

# Run benchmark
results_table = benchmark(
    models_path="path/to/model/pickles",
    display_names=display_names,
    metrics=metrics
)

print(results_table)
```

For a detailed example of how we generate the tables used in our paper, please see the [example notebook](https://github.com/Bizilizi/VGGSounder/blob/main/experiments/visualisations/metrics.ipynb).


## 📑 Citation

If you find VGGSounder useful for your research and applications, please consider citing us using this BibTeX:

```bibtex
@article{zverevwiedemer2025vggsounder,
  author    = {Daniil Zverev, Thaddäus Wiedemer, Ameya Prabhu, Matthias Bethge, Wieland Brendel, A. Sophia Koepke},
  title     = {VGGSounder: Audio-Visual Evaluations for Foundation Models},
  year      = {2025},
}
```

## ❤️ Acknowledgement
The authors would like to thank [Felix Förster](https://www.linkedin.com/in/felix-f%C3%B6rster-316010235/?trk=public_profile_browsemap&originalSubdomain=de), [Sayak Mallick](https://scholar.google.fr/citations?user=L_0KSXUAAAAJ&hl=en), and [Prasanna Mayilvahananan](https://scholar.google.fr/citations?user=3xq1YcYAAAAJ&hl=en) for their help with data annotation, as well as [Thomas Klein](https://scholar.google.de/citations?user=3WfC0yMAAAAJ&hl=en) and [Shyamgopal Karthik](https://scholar.google.co.in/citations?user=OiVCfscAAAAJ&hl=en) for their help in setting up MTurk. They also thank numerous MTurk workers for labelling. This work was in part supported by the [BMBF](https://www.bmbf.de/DE/Home/home_node.html) (FKZ: 01IS24060, 01I524085B), the [DFG](https://www.dfg.de/) (SFB 1233, TP A1, project number: 276693517), and the [Open Philanthropy Foundation](https://www.openphilanthropy.org/) funded by the [Good Ventures Foundation](https://www.goodventures.org/). The authors thank the IMPRS-IS for supporting TW.


## 👮 License

This project is released under the Apache 2.0 license as found in the LICENSE file. Please get in touch with us if you find any potential violations. 