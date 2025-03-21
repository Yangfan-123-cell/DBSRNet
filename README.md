# DBSRNet: Dual-Branch Network for No-Reference Super-Resolution Image Quality Assessment


This repository contains the official implementation of "Dual-Branch Network for No-Reference Super-Resolution Image Quality Assessment" as described in our paper.

## Abstract

No-reference super-resolution image quality assessment (SR-IQA) has become a critical technique for optimizing SR algorithms. The key challenge is how to comprehensively learn visual related features of SR images. Existing methods ignore the context information and feature correlation. To tackle this problem, we propose a dual-branch network for no-reference super-resolution image quality assessment (DBSRNet).

Our approach includes:
- **Dual-branch feature extraction module** - combining residual network and receptive field block net to learn multi-scale local features, while stacked vision transformer blocks learn global features
- **Correlation-based feature fusion** - learning and fusing correlations between dual-branch features using a self-attention mechanism structure
- **Adaptive feature pooling** - generating the final predicted score through an adaptive weighting strategy

Experimental results show that DBSRNet significantly outperforms state-of-the-art methods in terms of prediction accuracy on all SR-IQA datasets.

## Framework

![DBSRNet Framework](assets/framework.png)

## Installation

```bash
git clone https://github.com/Yangfan-123-cell/DBSRNet.git
cd DBSRNet
pip install -r requirements.txt
```

## Datasets

- **CVIU-17**: 1620 SR images generated by bicubic interpolation with eight SR algorithms and six scaling factors
- **QADS**: 980 SR images generated by 21 methods with three scaling factors
- **SISRSet**: 360 SR images generated by eight SR algorithms with three scaling factors (used for cross-validation)

## Train

Run the training script with default parameters:

```bash
python train.py
```

### Training Parameters

The default training configuration in `train.py`:

```python
# Training hyperparameters
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATCH_SIZE = 224
RANDOM_CROP = True
RANDOM_FLIP = True

# Optimizer settings
BETA1 = 0.9
BETA2 = 0.999

# Dataset split
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
```

### Training Process

The training procedure includes:
1. Loading and preprocessing the dataset (random cropping to 224×224 patches)
2. Data augmentation with random flipping
3. Model initialization with dual-branch architecture
4. Training with AdamW optimizer
5. Computing MSE loss between predicted scores and ground truth scores
6. Model evaluation using SROCC, PLCC, KROCC, and RMSE metrics

## Model Architecture

Our DBSRNet consists of three key modules:

### 1. Feature Extraction
- **Local-feature branch**: Combines ResNet50 (first three blocks) with RFBNet to extract multi-scale local features
- **Global-feature branch**: Uses stacked Vision Transformer blocks to extract global features

### 2. Feature Fusion
- Correlation analysis module based on self-attention mechanism
- Enhances intrinsic correlation between local and global features

### 3. Feature Pooling
- Adaptive weighting strategy for final score prediction

## Results

DBSRNet achieves state-of-the-art performance on SR-IQA datasets:

| Method    | CVIU-17 |       |       |       | QADS   |       |       |       |
|-----------|---------|-------|-------|-------|--------|-------|-------|-------|
|           | SROCC   | PLCC  | KROCC | RMSE  | SROCC  | PLCC  | KROCC | RMSE  |
| DBSRNet   | 0.9824  | 0.9845| 0.8867| 0.4206| 0.9835 | 0.9830| 0.8866| 0.4961|
| TADSRNet  | 0.9516  | 0.9585| 0.8120| 0.7966| 0.9720 | 0.9742| 0.8616| 0.6718|
| KDE-SRIQA | 0.9527  | 0.9513| 0.8044| 0.7618| 0.9632 | 0.9661| 0.8472| 0.6249|



## Acknowledgements

This work was supported by National Natural Science Foundation of China (62402074), the Major Special Project for Technological Innovation and Application Development in Chongqing (CSTB2024TIAD-STX0024), the Science and Technology Research Program of Chongqing Municipal Education Commission (KJQN202300632), Chongqing Postdoctoral Special Funding Project (2022CQBSHTB2057).
