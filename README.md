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
The authors would like to thank Felix Förster, [Sayak Mallick](https://scholar.google.fr/citations?user=L_0KSXUAAAAJ&hl=en), and [Prasanna Mayilvahananan](https://scholar.google.fr/citations?user=3xq1YcYAAAAJ&hl=en) for their help with data annotation, as well as [Thomas Klein](https://scholar.google.de/citations?user=3WfC0yMAAAAJ&hl=en) and [Shyamgopal Karthik](https://scholar.google.co.in/citations?user=OiVCfscAAAAJ&hl=en) for their help in setting up MTurk. They also thank numerous MTurk workers for labelling. This work was in part supported by the BMBF (FKZ: 01IS24060, 01I524085B), the DFG (SFB 1233, TP A1, project number: 276693517), and the Open Philanthropy Foundation funded by the Good Ventures Foundation. The authors thank the IMPRS-IS for supporting TW.


## 👮 License

This project is released under the Apache 2.0 license as found in the LICENSE file. Please get in touch with us if you find any potential violations.