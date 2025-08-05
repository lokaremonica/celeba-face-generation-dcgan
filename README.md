# 😎 Face Generation using DCGAN on CelebA (PyTorch)

This project implements a **Deep Convolutional GAN (DCGAN)** to generate realistic human face images based on the CelebA dataset. The generator learns to mimic facial features by mapping random noise vectors into high-resolution 64x64 images.

---
## ⚠️ Notebook Size Notice

The notebook file (`celeba-face-generation-dcgan.ipynb`) contains image outputs from multiple epochs and may be too large to preview directly on GitHub.Download & Run Locally or in Colab.

---
## 🧠 Model Overview

### 🧪 Dataset: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

* 200K celebrity face images, aligned and cropped
* Used 64x64 resolution subset
* Automatically downloaded using `torchvision.datasets.CelebA`

---

## ⚙️ Training Configuration

| Setting       | Value                    |
| ------------- | ------------------------ |
| Image Size    | 64x64 RGB                |
| Epochs        | 5                        |
| Latent Dim    | 100                      |
| Batch Size    | 128                      |
| Optimizer     | Adam (lr=0.0002, β1=0.5) |
| Loss Function | Binary Cross Entropy     |

---

## 🧠 Model Architectures

### Generator:

* Input: 100-dim noise vector
* 5 Transposed Conv layers with BatchNorm & ReLU
* Output: 64x64x3 image with `Tanh` activation

### Discriminator:

* Input: 64x64x3 image
* 5 Conv layers with LeakyReLU & BatchNorm
* Output: 1D probability (real or fake)

---

## 📊 Training Loss Curves

Plots of Generator and Discriminator loss over iterations(refer ipynb).

---

## 🖼️ Results – Generated Faces

Generated images from different epochs (refer ipynb)

---

## 🧠 Key Learnings

* GANs can learn facial structure and features from scratch
* Generator learns best when Discriminator isn't too overpowering
* BatchNorm and Tanh/ReLU stabilizes image generation
* Even 5 epochs can produce facial-like outputs from noise

---
