# SSL-BET: A Semi-Supervised Method for Brain Extraction on DWI Images

This repository contains the official implementation for the paper **"Automated brain extraction on diffusion-weighted images using pseudo and cross semi-supervised method"**.

SSL-BET is a semi-supervised learning framework designed to achieve high-accuracy brain extraction (skull-stripping) on Diffusion-Weighted Imaging (DWI) scans, particularly for stroke patients, even with very limited labeled data.

## 🔗 Model Weights

The trained model weights are hosted on the Hugging Face Hub for easy access and reproducibility. You can download them directly from the link below:

**Hugging Face Model Hub:** [https://huggingface.co/yibingchen/SSL-BET/tree/main](https://huggingface.co/yibingchen/SSL-BET/tree/main)

## 🚀 Training Guide

The following guide outlines the complete training pipeline for training on your own data.

### 1. Environment Setup

1.  Create and activate a Python 3.11 virtual environment.
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Data Preparation

Organize your data following the structure in the `example_data` folder:
- Each patient's data should be in its own subfolder.
- Place all patient subfolders into the `training`, `validation`, `test`, and `unlabeled` directories (you can customize these names).
  - **`training` and `validation`**: Must contain the corresponding ground truth labels.
  - **`test` and `unlabeled`**: Require only the image data; labels are not needed.

### 3. Training Pipeline

#### Step 1

3.  **Generate Config File**: Run the `create_config_baseline.ipynb` notebook to create the configuration file for the first step. The default output is `baseline_config.json`.
4.  **Train the Model**:
    ```bash
    python3 unet3d/scripts/train.py --config_filename baseline_config.json
    ```
    **Troubleshooting**: If you encounter a `ModuleNotFoundError`, add the project path to your `PYTHONPATH`:
    ```bash
    export PYTHONPATH="/path/to/your/project/root:$PYTHONPATH" (Change "/path/to/your/project/root" to you own path of unet3d folder)
    ```
5.  **Locate the Best Model**: After training, the best model (`model_best.pth`) is saved in the `baseline_config` folder. **This model is crucial for initializing subsequent steps.**

#### Step 2

6.  **Generate Config File**: Run the `create_config_step2.ipynb` notebook to generate `step2_config.json`.
7.  **Train the Model**:
    ```bash
    python3 unet3dStep2/scripts/train.py --config_filename step2_config.json --pretrained_model_filename "/path/to/step1/model_best.pth"
    ```
8.  **Generate Pseudo-Labels**:
    - **Prediction**:
      ```bash
      python3 unet3dStep2/scripts/predict.py --config_filename step2_config.json --model_filename "/path/to/step2/model_best.pth" --output_directory "/path/to/save/predictions" --activation sigmoid --group test
      ```
    - **Post-processing**: You need to **write a small script** to:
        1. Convert the prediction outputs into binary labels.
        2. Organize these labels into the `unlabeled` directory, matching the format of the `training` set.

#### Step 3

9.  **Generate Config File**: Run the `create_config_step3.ipynb` notebook to generate `step3_config.json`.
10. **Train the Final Model**:
    ```bash
    python3 unet3dStep3/scripts/train.py --config_filename step3_config.json --pretrained_model_filename "/path/to/step1/model_best.pth"
    ```
11. **Obtain the Result**: The final best model (`model_best.pth`) is saved in the `step3_config` folder. **This is the optimal model produced by the complete SSL-BET pipeline.**

## 📄 Citation

If you use SSL-BET in your research, please cite our paper:

```bibtex
@article{CHEN2026131825,
    title = {Automated brain extraction on diffusion-weighted images using pseudo and cross semi-supervised method},
    journal = {Neurocomputing},
    volume = {659},
    pages = {131825},
    year = {2026},
    issn = {0925-2312},
    doi = {https://doi.org/10.1016/j.neucom.2025.131825},
    url = {https://www.sciencedirect.com/science/article/pii/S092523122502497X},
    author = {Yibing Chen and Benqi Zhao and Jinyang Li and Zhilin Wang and Yingchun Fan and Kaiyue Su and Zhuozhao Zheng and Zhensen Chen},
    keywords = {Brain extraction, Skull-stripping, Semi-supervised learning, Stroke, Diffusion-weighted imaging},
    abstract = {Brain extraction in diffusion-weighted images (DWI) is a primary and crucial step in lesion analysis for stroke patients...}
}
