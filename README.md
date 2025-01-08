# StreetReview Project

## Overview
The StreetReview Project leverages the [StreetReview dataset](https://huggingface.co/datasets/rsdmu/streetreview) to develop an AI-based framework for assessing urban streetscape inclusivity. This repository contains code for:
1. **Feature Extraction**
2. **Model Training**
3. **Model Inference**

The project uses semantic segmentation, a multi-output neural network with attention mechanisms, and batch inference to predict inclusivity and accessibility scores from street-view images.

## Dataset
We utilize the [StreetReview dataset](https://huggingface.co/datasets/rsdmu/streetreview), which includes metadata, street-view images, and demographic evaluations. Visit the link for details.

## Project Structure
```
StreetReview/
├── .gitignore
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── 01_extract_features.py
│   ├── 02_train_model.py
│   └── 03_inference.py
├── data/
│   └── model_large.pth
└── notebooks/
    ├── heatmap_visualizations.ipynb
    └── radar_visualizations.ipynb
```

- **`data/model_large.pth`**: Pretrained model weights.
- **`notebooks/`**: Contains visualizations, including heatmaps and radar charts.
- **`environment.yml` & `requirements.txt`**: Define dependencies for Conda and pip.
- **`scripts/`**: Main Python scripts for feature extraction, model training, and inference.

## Model Description
The core model is a multi-output neural network with a **Multi-Head Attention** mechanism, processing feature logits from images to predict 28 evaluation metrics, such as inclusivity, accessibility, and aesthetics.

## Scripts Overview

### 1. `scripts/01_extract_features.py`
Processes street-view images to extract semantic segmentation logits using a Segformer model and saves them as CSV files.

**Usage:**
```bash
python scripts/01_extract_features.py
```

### 2. `scripts/02_train_model.py`
Defines and trains the multi-output model, logging training progress and saving the trained model.

**Usage:**
```bash
python scripts/02_train_model.py
```

### 3. `scripts/03_inference.py`
Performs batch predictions on new data using the trained model and saves results to a `predictions.csv` file.

**Usage:**
```bash
python scripts/03_inference.py
```

## Environment Setup

### Conda Environment
Use the provided `environment.yml` to set up:
```bash
conda env create -f environment.yml
conda activate streetreview_env
```

### Pip Installation
Alternatively, use pip:
```bash
pip install -r requirements.txt
```

## Additional Notes
- Ensure paths (e.g., model, data directories) are correctly configured.
- Scripts support GPU acceleration; they will fallback to CPU if necessary.
