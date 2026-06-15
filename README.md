# COLMBO-DF: Audio Language Model for Deepfake Detection

This repository contains my PyTorch implementation and reproduction pipeline for the state-of-the-art explainable deepfake speech detection framework: **COLMBO-DF**, based on the 2026 paper *"Audio Language Model for Deepfake Detection Grounded in Acoustic Chain-of-Thought"* (Carnegie Mellon University).

## 🚀 Project Overview
* **Architecture:** A reasoning-centric Audio-LLM pipeline that decouples acoustic representation learning from decision reasoning.
* **Audio Encoder:** Pretrained `WavLM-base-plus` (frozen).
* **Projector Module:** 6-layer `QFormer` network aligning audio representations into the LLM space.
* **LLM Backbone:** `Llama-3.2-1B-Instruct` as the central reasoning core.
* **Core Paradigm:** Integrating explicit, structured textual descriptions of low-level acoustic features directly into the text prompt to drive Feature-Grounded Chain-of-Thought (CoT) reasoning.

## 🛠️ Compute Environment
* **Platform:** Google Colab Pro+
* **Hardware Accelerator:** NVIDIA A100 Tensor Core GPU (Cloud Computing)
* **Frameworks:** PyTorch, Hugging Face `transformers`, `accelerate`, `torchaudio`.

## 📌 Critical Finding & Reproduction Focus
As highlighted in the original paper's ablation studies, deepfake speech detection within this LLM framework relies heavily on **explicit acoustic evidence** (including pitch, formants, energy, jitter, spectral statistics, and silence tracking). 

Removing these explicit acoustic fields causes the training to collapse into a degenerate solution (chance-level accuracy). Therefore, this pipeline carefully preserves and structures the serialized text blocks of low-level acoustic features alongside the projected audio embeddings to guarantee stable convergence and robust decision rationales.

## 📈 Current Status
* [x] Repository initialized and structural codebase (`train.py`) pushed.
* [x] Environment dependencies configured for automated cloud execution.
* [x] Active training pipeline tracking loss convergence and output schemas in `training_log.txt`.
