# Pragmatic_Ablities_of_VLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Phil. Trans. R. Soc. B](https://img.shields.io/badge/Paper-RSTB--2026-green.svg)](https://royalsocietypublishing.org/journal/rstb)

Official repository for the paper:
> **Interactive Abilities of Visually Grounded Conversational Models in the LLM Era**  
> *Samuele Antonelli, Raffaella Bernardi, Raquel Fernández*  
> Philosophical Transactions of the Royal Society B (RSTB), 2026.

The full supplementary material is in [`appendix.md`](appendix.md).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Experiments](#running-experiments)
- [Main Experimental Results](#main-experimental-results)
- [Citation](#citation)
- [License](#license)

---

 
## Overview

Human conversation is a joint activity where speakers coordinate through efficiency principles, balancing conciseness and informativeness (Grice's Maxim of Quantity, Clark's Principle of Least Collaborative Effort). While multimodal large language models (VLMs) demonstrate strong vision-language capabilities, their interactive coordination abilities remain poorly understood.

This repository provides an empirical evaluation framework combining two complementary paradigms:

1. **Behavioral Evaluation (Referring Expression Comprehension, REC):** Tests whether model listeners (principally **Qwen2.5-VL-7B**, alongside **CogVLM-17B** and **Grounding DINO-L** for comparison) benefit from human-like concise descriptions compared to verbose model-generated descriptions under varying visual clutter (0 to 4 same-category distractors).
2. **Representational Probing:** Linear classifiers trained on hidden states across layers (focused on layer 18) to determine whether pragmatic discriminativity is linearly decodable, with:
   - **Text-Only Ablation:** how much decodability comes from surface linguistic regularities alone.
   - **Random Labels Control:** a chance-level baseline (AUROC ≈ 0.50).
   - **McNemar Discrepancy Test:** whether visual representations correct or introduce unimodal errors.

---

## Repository Structure

```text
Pragmatic_Ablities_of_VLMs/
├── README.md
├── appendix.md                      # Supplementary material (Appendices A–E)
├── LICENSE
├── .gitignore
├── requirements.txt
├── behavioral/
│   └── eval_rec.py                  # REC inference and IoU computation (0.50, 0.75, mean IoU)
├── representational/
│   └── probe_pipeline.py            # Last-token extraction across layers, linear probes, random-label control
├── analysis/
│   └── error_and_linguistic.py      # Lexical cue distribution and McNemar cross-modal test
└── supplementary/                   # Reviewer-response analyses (see supplementary/README.md)
    ├── robustness/
    ├── distractor_stratification/
    ├── error_analysis/
    └── control_baselines/
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/samueleantonelli/Pragmatic_Ablities_of_VLMs.git
cd Pragmatic_Ablities_of_VLMs
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Running Qwen2.5-VL-7B requires a CUDA GPU with enough VRAM and `flash_attention_2` support.

---

## Data Preparation

Experiments use the **RefOI** benchmark (Ma et al., 2025), available on Hugging Face as [`Seed42Lab/RefOI-TLHF`](https://huggingface.co/datasets/Seed42Lab/RefOI-TLHF), [`Seed42Lab/RefOI`](https://huggingface.co/datasets/Seed42Lab/RefOI).

The human listener annotations (discriminative vs. underdiscriminative labels) were provided by the original RefOI authors.[^1]

[^1]: Note: the human annotations were obtained from the authors of this paper: https://arxiv.org/abs/2504.16060.

1. Download the dataset from Hugging Face and place the images in `data/images/`.
2. Build splits partitioned strictly by `image_id` to prevent visual leakage:
   - **Training (~70%):** 9,175 pairs
   - **Validation (~15%):** 1,961 pairs
   - **Test (~15%):** 1,915 pairs

   Overall, the probing corpus has 5,341 discriminative and 7,710 underdiscriminative instances.

Expected format for `data/splits/test_rec.csv`:
```csv
image_id,image_file,speaker,condition,distractor_count,gold_box,description
1042,dog_scene_1.jpg,Human,Spoken,2,"[63, 184, 695, 907]","The black dog in the front."
```

---

## Running Experiments

### 1. Behavioral Evaluation (REC)

Prompt the listener to output bounding box coordinates `[Xmin, Ymin, Xmax, Ymax]` and compute IoU against gold standards:

```bash
python behavioral/eval_rec.py \
    --data_path data/splits/test_rec.csv \
    --images_dir data/images/ \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_path rec_results.csv
```

Outputs: accuracy at IoU ≥ 0.50 and IoU ≥ 0.75, mean IoU, and breakdowns across 0 to 4 same-category distractors.

### 2. Representational Probing

Extract last-token hidden states across layers without a conversational prompt template, and train a class-balanced linear classifier on discriminativity:

```bash
python representational/probe_pipeline.py \
    --data_csv data/splits/probing_dataset.csv \
    --images_dir data/images/ \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct"
```

The layer is selected on the **validation split only** (highest validation AUROC over all even layers and layer 27); the per-layer validation scores are written to `layer_selection_val.csv`. The test split is not used for this choice. Pass `--layer 18` to fix the layer instead.

This script runs:
1. **Multimodal probing:** hidden states from (Image + Text) at the selected layer.
2. **Text-Only ablation:** hidden states without the image.
3. **Random-label control:** permuted training labels, to test chance level.

### 3. Linguistic Analysis & McNemar Test

Quantify spatial terms, ordinals, definite articles, colour terms and sentence length, and compute the McNemar test on cross-modal discrepancies:

```bash
python analysis/error_and_linguistic.py \
    --test_results_csv test_probe_predictions.csv
```

---

## Main Experimental Results

### Listener Comprehension Accuracy (IoU ≥ 0.50) [^2]:

| Speaker | Condition | Human Listeners | Qwen2.5-VL-7B (Full) | Qwen2.5-VL-7B (≥ 1 Distractor) | CogVLM-17B | Grounding DINO-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Human** | Spoken | 92% | 88.31% | **85.35%** | 63% | 60% |
| **Human** | Written | 89% | 88.15% | **84.90%** | 60% | 60% |
| **Model** | Brief | 36% | 88.19% | 82.33% | 61% | 73% |
| **Model** | Default | 40% | 88.72% | 81.56% | 37% | 58% |

[^2]: Note: the human-as-listener results were obtained from Ma et al., 2026 (https://arxiv.org/abs/2504.16060) and reported here for the sake of comparison.

### Linear Probing at Layer 18 (Qwen2.5-VL-7B)

| Probing Condition | Accuracy | F1 Score | AUROC |
| :--- | :---: | :---: | :---: |
| **Multimodal (Image + Text)** | **80%** | **0.78** | **0.89** |
| **Unimodal (Text-Only Ablation)** | 74% | 0.73 | 0.84 |
| **Control (Random Labels)** | 49% | 0.43 | 0.50 |

### Key Takeaways

1. **Modest sensitivity to description condition:** Qwen2.5-VL-7B keeps relatively high accuracy across description sources and prompt styles, unlike human listeners, whose accuracy drops to 36–40% on model descriptions. The sensitivity is clearer with same-category competitors present (≥ 1 distractor), where human descriptions have a small but significant advantage over Default prompts (p = 0.006). Human and model descriptions differ in several respects, so this comparison does not isolate the effect of conciseness.
2. **Text-only probing is strong:** Unimodal probing reaches AUROC 0.84 without the image, against 0.89 with it. Much of the probe's performance can therefore be explained by surface textual regularities, although the image adds information.
3. **Visual features correct more errors than they introduce:** On the 265 trials where the multimodal and text-only probes disagree, the multimodal probe is right in 183 cases and wrong in 82 (McNemar χ² = 38.5, p < 0.001). This pattern is consistent with a corrective role for visual information in the representations, but it does not establish how the model uses that information during generation.

Further results, including the robustness and control analyses, are in [`appendix.md`](appendix.md).

---

## Citation

If you find this work or codebase useful, please cite our paper:

```bibtex
@article{antonelli2026interactive,
  title={Interactive Abilities of Visually Grounded Conversational Models in the LLM Era},
  author={Antonelli, Samuele and Bernardi, Raffaella and Fern{\'a}ndez, Raquel},
  journal={Philosophical Transactions of the Royal Society B},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
