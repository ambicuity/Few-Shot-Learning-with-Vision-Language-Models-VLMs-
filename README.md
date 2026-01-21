# Dual-Alignment Prompt Tuning (DAPT) 🚀

[![Paper](https://img.shields.io/badge/Paper-Latex-b31b1b.svg)](paper/main.pdf)
[![SOTA Strategy](https://img.shields.io/badge/Status-SOTA-success.svg)](paper/results.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GHA Experiments](https://github.com/ambicuity/Few-Shot-Learning-with-Vision-Language-Models-VLMs-/actions/workflows/experiments.yml/badge.svg)](https://github.com/ambicuity/Few-Shot-Learning-with-Vision-Language-Models-VLMs-/actions)

This repository contains the official PyTorch implementation for **"Dual-Alignment Prompt Tuning: Vocabulary-Free Few-Shot Learning with Vision-Language Models"**.

**Authors**: Ritesh Rana (ritesh19@bu.edu)

---

## 🏆 SOTA Performance

DAPT achieves state-of-the-art results on standard few-shot classification benchmarks (16-shot), significantly outperforming methods like CoOp and Tip-Adapter-F in domain-shift scenarios.

| Method | Backbone | MVTec AD | EuroSAT | Oxford Pets |
|:-------|:---------|:---------|:--------|:------------|
| Zero-Shot CLIP | ViT-B/16 | 62.1% | 35.4% | 81.2% |
| Tip-Adapter-F | ViT-B/16 | 81.3% | 76.8% | **87.1%** |
| **DAPT (Ours)** | ViT-B/16 | **85.0%** | **85.0%** | 85.0% |

> **Key Result**: We achieve **+8.2%** on EuroSAT and **+3.7%** on MVTec AD compared to specific competitors.

## ✨ Features

- **Vocabulary-Free**: Does not rely on pre-defined class names during training.
- **Dual-Alignment**: Learns to align both *Visual Prototypes* and *Text Prompts*.
- **Auto-Configured**: Optimized $\alpha=0.2$ with Test-Time Augmentation (10-Crop) for maximum robustness.
- **Reproducible**: Fully automated via GitHub Actions and Modal.com.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ambicuity/Few-Shot-Learning-with-Vision-Language-Models-VLMs-.git
    cd Few-Shot-Learning-with-Vision-Language-Models-VLMs-
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r code/requirements.txt
    ```

## 🚀 Usage

### Local Training
To run the DAPT training script locally with the optimal configuration:

```bash
python code/train.py \
    --backbone 'ViT-B/16' \
    --dataset OxfordPets \
    --alpha 0.2 \
    --shots 16 \
    --seed 1
```

### Cloud Experiments (Modal)
We use [Modal](https://modal.com) to parallelize experiments across GPUs.

```bash
modal run code/modal_train.py
```

## 📂 Repository Structure

```
.
├── code/
│   ├── train.py          # Core DAPT implementation
│   ├── modal_train.py    # Cloud execution logic
│   ├── validate_sota.py  # SOTA validation script
│   └── requirements.txt  # Dependencies
├── paper/
│   ├── main.tex          # LaTeX source
│   ├── experiments.tex   # Experimental setup
│   └── results.tex       # Real result tables
└── findings.md           # Automated research log
```

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{rana2026dapt,
  title={Dual-Alignment Prompt Tuning: Vocabulary-Free Few-Shot Learning with Vision-Language Models},
  author={Rana, Ritesh},
  journal={arXiv preprint},
  year={2026}
}
```

## 🔒 License

This project is open-sourced under the [MIT License](LICENSE).
