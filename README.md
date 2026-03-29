# Image Colorization Using Deep Convolutional Neural Networks

## Overview
This project addresses the challenging computer vision task of **Image Colorization**. The core objective is to predict plausible color information for grayscale images to render them realistically. Because a single black-and-white pixel can correspond to multiple real-world colors, this project utilizes Deep Convolutional Neural Networks (DCNNs) within an encoder-decoder framework.

The model extracts deep features from the input luminance (L) channel using the encoder, and uses these features in the decoder to reconstruct the missing color (ab) channels within the CIELAB color space.

## Author
* **Name:** Alihan Ertekin
* **Course:** BBM 418 & AIN 433 Computer Vision Laboratory (Assignment 3)

## Dataset Preparation
A custom PyTorch Dataset class (`MYColorizationDataset`) is implemented to handle data loading and preprocessing efficiently:
* Images are loaded directly into RAM to ensure rapid access during the training process.
* Images are resized to **256x256 pixels** using nearest-neighbor interpolation.
* The color space is transformed from **RGB to Lab** to separate lightness from color components.
* **Normalization:** The `L` channel is normalized by dividing by 100.0, and the `a` and `b` channels are normalized by dividing by 128.0.
* The `L` channel is fed into the model as input, while the `ab` channels serve as the target prediction.

## Implementation Details
* **Framework:** PyTorch
* **Computer Vision Library:** **Kornia** is heavily utilized for this project due to its seamless compatibility with PyTorch. It allows image transformations (like color space conversions) to be performed directly on the GPU (CUDA), increasing computational efficiency and keeping the training pipeline fully differentiable.
* **Metrics:** **TorchMetrics** is used for robust performance evaluation.

## Model Architectures
Two different network architectures are implemented and compared:

### 1. Baseline Model
This model strictly replicates a base encoder-decoder architecture. The encoder uses sequential convolutional layers followed by Adaptive Average Pooling to downsample feature maps and increase the receptive field. 
To capture high-level semantic context, a **Global Feature Extractor** is integrated. It utilizes either a pre-trained **EfficientNet-B0** or a custom CNN with Adaptive Max Pooling. The local features from the encoder are concatenated with the global feature vector before passing through the decoder to restore spatial resolution.

### 2. Improved Model (U-Net Based)
To address potential vanishing/exploding gradients and enhance visual coherence, a **U-Net-based architecture** is implemented as an improvement over the baseline. Direct skip connections are introduced between the encoder and decoder layers. By concatenating feature maps from the encoding path directly with the decoding path, the model prevents the loss of high-resolution spatial information.

## Loss Functions
To enforce texture consistency and high realism in the generated images, the following losses are utilized:
* **L1 Loss:** For pixel-level reconstruction.
* **Perceptual Loss:** Implemented using a frozen, pre-trained **VGG16** network (ImageNet weights). Instead of pixel differences, Mean Squared Error (MSE) is calculated between the high-level feature representations of the predicted and target RGB images, scaled by a lambda factor.

## Evaluation Metrics
The models are evaluated using standard image reconstruction metrics:
* **MSE** (Mean Squared Error)
* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index Measure)

## Requirements & Usage
The project is designed to be run on Google Colab with Google Drive integration. To run the notebook locally or on another environment, ensure the following dependencies are installed:

```bash
pip install torch torchvision kornia torchmetrics scikit-learn pandas matplotlib
