# E9-222: Signal Processing in Practice

Course assignments from **E9-222: Signal Processing in Practice** at the **Indian Institute of Science (IISc), Bangalore** (2026).

Each assignment is implemented as a Jupyter notebook with a companion PDF report.

---

## Assignments

### Assignment 1: Linear and Circular Convolution
Implements convolution from scratch using three approaches: manual loops, Toeplitz matrix algebra, and FFT-based circular convolution. Applies these to audio de-reverberation (Wiener filtering) and 2D image deblurring with motion and Gaussian blur kernels, including blind deconvolution via grid search.

**Dataset:** Audio samples from [SigSRP / Signalogic Engineering Samples](https://www.signalogic.com/melp/EngSamples/)

---

### Assignment 2: Discrete Cosine Transform (DCT)
Constructs 1D and 2D DCT basis functions manually and applies block-based DCT for JPEG-style image compression. Explores energy compaction, quantization effects, and the quality-vs-compression tradeoff on the Set5 benchmark.

**Dataset:** [Set5 Super-Resolution Dataset](https://github.com/jbhuang0604/SelfExSR) (auto-downloaded in notebook)

---

### Assignment 3: Audio Denoising and Dereverberation
Builds a deep learning pipeline (CNN-based) to denoise and dereverberate speech signals. Uses dysarthric and healthy speaker recordings corrupted with five noise types (white, babble, pink, machinery, factory) and room impulse responses. Evaluates performance with standard speech quality metrics.

**Dataset:** Dysarthric and healthy speech corpus — provided by the course (internal dataset, not publicly available). Noise and RIR files included in `Noises&RIR/`.

---

### Assignment 4: Linear Prediction (LP) Analysis
Implements LP coefficient estimation using the autocorrelation method and the Levinson-Durbin algorithm. Analyzes residual energy for voiced/unvoiced frame detection, computes prediction gain, and explores LP spectral envelopes. Also investigates the role of phase in speech synthesis and power-law LP spectrum fitting.

**Dataset:** Four 16 kHz WAV files (2 speakers × 2 utterances) provided by the course (`dataFiles/`).

---

### Assignment 5: Spectral Estimation and Robust Optimization
Applies spectral estimation techniques and robust optimization methods to image processing tasks. Covers frequency-domain filtering and optimization-based signal reconstruction.

**Dataset:** Test images provided by the course (`imagesforlab1/`).

---

### Assignment 6: Image Restoration and Filtering
Explores classical image restoration techniques including Gaussian filtering, bilateral filtering, edge-preserving smoothing, and adaptive filtering. Compares filter performance on degraded images.

**Dataset:** Test images provided by the course (`imagesforA11/`).

---

### Assignment 7: Incremental Learning — Pet Classification
Trains a ResNet-18 from scratch on a 20-class base set of the Oxford-IIIT Pet Dataset, then incrementally extends it to 17 new classes. Studies catastrophic forgetting and mitigates it via synthetic feature generation and feature distribution replay.

**Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds, ~7,400 images.

---

### Assignment 8: Data Augmentation — MixUp and Manifold MixUp
Trains a custom SimpleCNN on CIFAR-10 and benchmarks: (1) no augmentation, (2) standard augmentations (RandomCrop, ColorJitter, Gaussian noise), (3) input-level MixUp (α ∈ {0.1, 0.2, 0.4, 1.0}), and (4) Manifold MixUp (mixing at intermediate layers k = 1, 2, 3, or random). Compares test accuracies across all configurations.

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — auto-downloaded via `torchvision.datasets`.

---

### Assignment 9: Semi-Supervised Learning — FixMatch
Implements FixMatch on a 3-class CIFAR-10 subset (cat, deer, dog) with only 12 labeled examples and ~5,400 unlabeled examples. Sweeps over confidence threshold τ ∈ {0.70, 0.80, 0.90, 0.95} and unsupervised loss weight λ ∈ {0.5, 1.0, 2.0}. Best configuration (τ=0.90, λ=0.5) achieves 71.37% test accuracy vs. 57.50% supervised baseline.

**Dataset:** CIFAR-10 3-class subset — derived from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) via `torchvision`; split files stored in `cifar10_3class_fixmatch/`.

**Reference:** [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (Sohn et al., 2020)](https://arxiv.org/abs/2001.07685)

---

### Assignment 10: Image Deblurring with Hessian Mixed Regularization
Solves image deblurring via ADMM using a mixed Hessian regularizer combining nuclear norm and Frobenius norm penalties: `R(M) = rw·‖M‖_* + (1−rw)·‖M‖_F`. Implements the proximal operator for this penalty applied per-pixel to 2×2 Hessian matrices. Sweeps over λ ∈ {0.0001, 0.005, 0.01, 0.015, 0.02} and rw ∈ {0, 0.5, 0.7, 1}.

**Dataset:** Two test images provided with the assignment (`Test_Image_1.jpg`, `Test_Image_2.jpeg`).

---

## Repository Structure

```
E9-222-Signal-Processing-in-practise/
├── Assignment_1/       # Convolution & deblurring
├── Assignment_2/       # DCT & image compression
├── Assignment_3/       # Audio denoising & dereverberation
├── Assignment_4/       # Linear prediction analysis
├── Assignment_5/       # Spectral estimation & optimization
├── Assignment_6/       # Image restoration & filtering
├── Assignment_7/       # Incremental learning (pet classification)
├── Assignment_8/       # MixUp & Manifold MixUp augmentation
├── Assignment_9/       # Semi-supervised learning (FixMatch)
└── Assignment_10/      # Image deblurring (Hessian regularization)
```

Each folder contains:
- `Assignment_N.ipynb` — Main implementation notebook
- `E9_*_report.pdf` — Submitted report
- Reference PDFs and slides (where applicable)

---

## Requirements

- Python 3.8+
- PyTorch, torchvision
- NumPy, SciPy, Matplotlib
- librosa (for audio assignments)
- scikit-learn, scikit-image
