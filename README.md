# Deep Learning Experiment README

# Project Overview

This repository documents the implementation, training configuration, dataset resources and experimental results of our deep learning research experiments. All related raw data, processed datasets, trained model weights and experimental output files are shared via Baidu Netdisk for reproducibility and resource reference.

---

## 📁 Resource Access

All experiment-related datasets and result files are packaged in a shared folder, which can be accessed via the following link:

**Shared Folder Name**: data&result
**Baidu Netdisk Link**: [https://pan.baidu.com/s/16Q_U6_nbUipMruPWf1ZDlQ](https://pan.baidu.com/s/16Q_U6_nbUipMruPWf1ZDlQ)
**Extraction Code**: hk99

The folder contains raw datasets, preprocessed data files, trained model checkpoints, training logs, evaluation metrics and visualization results of the experiments.

---

## ⚙️ Experiment Environment & Configuration

### Hardware Environment

- **GPU**: NVIDIA GeForce RTX 4090D (24GB dedicated memory)

- All network training and inference tasks are performed on this GPU device to ensure computational efficiency and speed.

### Software Environment

- **Deep Learning Framework**: PyTorch 2.0.0

- **CUDA Version**: 11.8 (for GPU acceleration support)

### Training Hyperparameters & Settings

- **Input Image Preprocessing**: Input images are cropped and resized to a uniform resolution of **224×224** to optimize computational efficiency and maintain feature consistency.

- **Optimizer**: Adam optimizer is adopted for network parameter updating, which delivers stable convergence and adaptive learning rate performance.

- **Initial Learning Rate**: 1e-4 (batch learning rate)

- **Batch Size**: 8 (adjusted to match GPU memory capacity and training stability)

- **Data Augmentation**: To alleviate overfitting caused by limited dataset size, common augmentation strategies are applied during data loading, including random scaling, random cropping and horizontal flipping.

- **Training Epochs**: The network is trained for a total of **150 epochs** to achieve full convergence.

- **Model Saving Strategy**: Model parameters are automatically saved when the optimal validation performance is achieved, retaining the best-performing checkpoint for subsequent evaluation and inference.

---

## 📊 Experimental Reproduction Guide

1. Download the `data&result` folder from the Baidu Netdisk link above, and unzip it to the local project root directory.

2. Configure the consistent software environment: install PyTorch 2.0.0 and matching CUDA 11.8 dependencies, and complete other auxiliary library installations.

3. Load the preprocessed dataset and the saved optimal model checkpoint for training reproduction or direct inference testing.

4. Keep the hyperparameters and preprocessing settings consistent with the configuration described above to replicate the experimental results accurately.

---

## ⚠️ Notes

- Ensure the GPU driver supports CUDA 11.8 to avoid compatibility issues during training and inference.

- Do not modify the data structure and file naming rules in the shared `data&result` folder to ensure normal loading of datasets and model weights.

- If the GPU memory is insufficient, appropriately reduce the batch size (adjust based on actual device conditions) while keeping other hyperparameters unchanged.

- The data augmentation module is integrated into the data loading pipeline, and no additional configuration is required for routine reproduction.
