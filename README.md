# DreamBooth Fine-Tuning Practice with Stable Diffusion v1.5 + LoRA

This repository contains a compact DreamBooth-style fine-tuning pipeline for subject-driven image generation using **Stable Diffusion v1.5** and **LoRA**. The project was developed to complete the full workflow of **data collection → preprocessing → fine-tuning → generation → analysis** on a small personalized subject dataset.

The subject used in this project is a **dragon plush toy**, represented during training with the identifier-token prompt:

> `a photo of sks dragon toy`

The project focuses on a controlled comparison between two LoRA configurations:

- **Setting A:** `Rank = 128`
- **Setting B:** `Rank = 16`

The purpose of this comparison is to analyze how LoRA capacity affects **subject fidelity**, **prompt fidelity**, **diversity**, and **prior preservation** in DreamBooth personalization.

---

## Project Overview

DreamBooth learns a specific visual identity from a very small set of instance images by associating a rare identifier token with a subject class. In this project, LoRA is used as a parameter-efficient alternative to full fine-tuning.

This repository includes:

- an **interactive preprocessing script** for preparing training images,
- a **training script** for DreamBooth + LoRA fine-tuning,
- an **inference and evaluation script** for controlled generation,
- a **loss plotting script** for visualizing training dynamics.

---

## Model and Dataset Release

### Trained LoRA Models
The LoRA models trained with the two settings are available at:

- **Hugging Face Model Repo:**  
  [https://huggingface.co/qqz03/DreamBooth-SDv1.5-LoRA](https://huggingface.co/qqz03/DreamBooth-SDv1.5-LoRA)

This model repository includes both trained configurations:

- **Config A:** LoRA Rank = 128
- **Config B:** LoRA Rank = 16

### Data Release
The original dataset, preprocessed dataset, and generated image results are available at:

- **Hugging Face Dataset Repo:**  
  [https://huggingface.co/datasets/qqz03/DreamBooth-SDv1.5-data](https://huggingface.co/datasets/qqz03/DreamBooth-SDv1.5-data)

This dataset repository contains:

- the **original training images**,
- the **preprocessed 512×512 training images**,
- the **200 class images** used for prior preservation.

---

## Repository Structure

```text
.
├── resize.py        # Interactive preprocessing and crop review
├── train.sh         # DreamBooth + LoRA training script
├── inference.py     # Controlled image generation and diversity evaluation
├── plot.py          # Training loss parsing and plotting
└── README.md
