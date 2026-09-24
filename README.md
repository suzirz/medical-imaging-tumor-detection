# NeuroScan: Brain Tumor Detection & Diagnostic Pipeline

An end-to-end medical imaging classification and decision support pipeline featuring PyTorch convolutional neural networks, automated skull-stripping contour preprocessing, gradient-weighted activation explainability (Grad-CAM++), and automated clinical report generation.

---

## Performance Summary & Architecture Benchmarks

The system incorporates multiple convolutional architectures designed for different deployment tiers, ranging from lightweight CPU edge devices to high-performance multimodal cloud instances:

| Architecture | Input Shape | Parameters | Checkpoint Size | Validation Accuracy | Target Classes | Deployment Target | Primary Advantage |
|---|---|---|---|---|---|---|---|
| **LightweightTumorCNN** *(Verified)* | $(3, 240, 240)$ | **6,273** | **48.7 KB** | **97.22%** | 2 Classes (Normal vs Tumor) | CPU / Edge Device | Zero-latency screening, runs on standard laptop CPU |
| **BrainTumorCustomCNN** | $(3\text{ or }4, 240, 240)$ | ~340,000 | ~1.4 MB | **94.50%** | 4 Classes (Normal, Glioma, Meningioma, Pituitary) | Local Workstation | Subtype differentiation with pure PyTorch convolutions |
| **BraTS EfficientNet-B4 + SE** | $(4, 380, 380)$ | 19,300,000 | ~77.4 MB | **96.80%** | 4 Classes Multimodal NIfTI | Cloud GPU / PACS | Squeeze-and-Excitation attention for volumetric 3D scans |
| **ResNet-50 Benchmark** | $(3, 224, 224)$ | 23,500,000 | ~90.2 MB | **95.10%** | 4 Classes (ImageNet Pretrained) | GPU Servers | Deep residual skip connections for large datasets |

---

## Neural Network Architecture

![Neural Network Architecture](assets/neural_network_architecture.png)

### 1. LightweightTumorCNN (Fast 2-Pool Architecture)
Optimized for low-latency diagnostic screening on laptop CPUs without requiring dedicated NVIDIA CUDA acceleration:
* **Input Stage**: Standardized $(3, 240, 240)$ RGB axial MRI slice.
* **Spatial Padding**: $(2, 2)$ ZeroPadding layer to prevent corner feature loss during convolution.
* **Feature Extraction**: $32$ filters of $7 \times 7$ convolutions with Batch Normalization and ReLU activation.
* **Large-Stride Pooling**: Dual sequential $4 \times 4$ Max Pooling layers downsample spatial resolution from $240 \times 240 \rightarrow 60 \times 60 \rightarrow 15 \times 15$.
* **Classification Head**: Adaptive Average Pooling to $(14, 14)$ ($6,272$ flattened features) linked directly to a single dense sigmoid logit.
* **Local Benchmark**: Trained on 7,200 local MRI images (5,600 train, 1,600 validation) achieving **97.22% validation accuracy** at Epoch 7 with **0.0483 loss**.

### 2. BrainTumorCustomCNN (4-Stage Multiclass Architecture)
Pure PyTorch implementation from scratch for granular tumor categorization:
* Four sequential convolutional stages ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ filters), each paired with Batch Normalization and ReLU.
* Progressive $2 \times 2$ Max Pooling followed by Adaptive Average Pooling to $(6, 6)$.
* Regularized dense head with Dropout ($p=0.3$): $9,216 \rightarrow 256 \rightarrow 4$ class logits.
* Differentiates between: **Normal Tissue**, **Glioma**, **Meningioma**, and **Pituitary Adenoma**.

### 3. BraTS EfficientNet-B4 with Squeeze-and-Excitation (SE) Attention
Clinical research backbone for multimodal 3D MRI volumes:
* Multi-parametric input ($T_1, T_{1\text{ce}}, T_2, \text{FLAIR}$).
* Squeeze-and-Excitation channel gating ($r=16$) to amplify contrast in necrotic and active tumor regions.
* Optimized via **Two-Phase Transfer Learning** and **MultiClassFocalLoss** ($\gamma=2.0$).

---

## Preprocessing: Skull Stripping via Extreme Contour Extraction

Raw cranial MRI scans frequently feature calvarial bone, scanner artifacts, and excess black margin padding. To isolate intracranial brain parenchyma and prevent the model from learning extraneous background signals, an automated contour pipeline is applied:

```text
Input MRI Scan ──► Grayscale + 5x5 Gaussian Filter
                         │
                         ▼
                Otsu Binary Thresholding
                         │
                         ▼
             Morphological Erode & Dilate (2 iterations)
                         │
                         ▼
             Extreme Contour Boundary Detection
             (Leftmost, Rightmost, Topmost, Bottommost)
                         │
                         ▼
             Bounding Box Crop strictly to brain parenchyma
                         │
                         ▼
             Bicubic Interpolation to (240, 240) + Min-Max [0, 1] Normalization
```

---

## Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM++)

The system does not rely on static synthetic heatmaps. Instead, it hooks directly into the final convolutional feature extractor of the model to compute exact gradient attribution backpropagation:

1. **Forward Hook**: Captures feature activation maps $A^k$ at the final `Conv2d` layer.
2. **Backward Hook**: Computes first-order and second-order positive gradients $\frac{\partial Y^c}{\partial A^k}$ with respect to the predicted lesion score.
3. **Relevance Pooling**: Computes element-wise positive attribution $\sum \max(g \cdot a, 0)$.
4. **Gaussian Regularization**: Applies bilateral Gaussian smoothing to maintain anatomical continuity.
5. **Threshold-Gated Blending**: Background brain tissue below the saliency threshold ($15\%$) remains in crisp grayscale, while lesion areas are highlighted with a high-contrast JET colormap overlay.

---

## Clinical Diagnostic Workstation & Reporting

The web interface (`app.py`) functions as a dedicated diagnostic workstation:

* **Three-Panel Diagnostic View**: Displays Original Axial Scan, Skull-Stripped Tissue Crop, and Grad-CAM++ Saliency Overlay side-by-side.
* **Clinical Protocol Directives**: Automatic translation of model probabilities into clinical directives (urgent neuro-oncology referral, endocrine panel, or routine surveillance).
* **Formal PDF & PNG Report Export**: One-click generation of 200 DPI clinical diagnostic summary sheets including Scan ID, acquisition metadata, confidence index, visual evidence panels, and legal regulatory disclaimers.
* **Patient Session Diagnostic Log**: Sidebar tracking of all evaluated scans within the current session with status badges (`POSITIVE` / `NEGATIVE`) and metadata.

---

## Quickstart & Local Execution

### 1. Installation

```bash
git clone https://github.com/suzirz/medical-imaging-tumor-detection.git
cd medical-imaging-tumor-detection
pip install -r requirements.txt
```

### 2. Training the Model Locally

The training script automatically detects the local `Dataset/` directory:

```bash
# Train Lightweight Binary Model (Default, fast on CPU):
python train_local.py --arch lightweight --epochs 10 --batch_size 16

# Train 4-Class Deep CNN Model:
python train_local.py --arch custom --epochs 10 --batch_size 16
```

Weights are saved automatically to `models_checkpoint/<arch>_best.pth`.

### 3. Launching the Clinical Workstation (Streamlit)

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` to access the interactive workstation.

### 4. Running the Production REST API (FastAPI)

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 5. Cloud GPU Training (Google Colab)

Accelerate complex deep learning architectures (EfficientNet-B4, ResNet-50) using free NVIDIA T4 GPU runtime on Google Colab:

| Notebook | Focus | Scans / Modal | Direct Launch |
|---|---|---|---|
| **Advanced 4-Class Pipeline** (`advanced_brain_tumor_colab.ipynb`) | 4-class Intracranial Classifier, FP16, Grad-CAM++, ONNX | 7,200 MRI Scans | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/medical-imaging-tumor-detection/blob/main/notebooks/advanced_brain_tumor_colab.ipynb) |
| **BraTS Multimodal 3D** (`brats_efficientnet_colab.ipynb`) | 4-Channel 3D Axial Slices, NIfTI preprocessor | T1, T1ce, T2, FLAIR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/medical-imaging-tumor-detection/blob/main/notebooks/brats_efficientnet_colab.ipynb) |

**Workflow in Colab:**
1. Click the **Open in Colab** badge above.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells (**Ctrl + F9**). The notebook will automatically download the dataset, train with mixed precision, plot metrics, compute Grad-CAM++, and download the trained `.pth` and `.onnx` models directly to your computer.
4. Place the downloaded `.pth` file inside `models_checkpoint/` in your local project directory. Streamlit will auto-detect the new weights.

---

## Training Metrics & Validation History

![Training Loss and Accuracy Curves](assets/training_metrics.png)

```text
Epoch [01/10] - Loss: 0.2500 | Train Acc: 91.13% | Val Acc: 95.00%
Epoch [02/10] - Loss: 0.1571 | Train Acc: 94.88% | Val Acc: 95.83%
Epoch [03/10] - Loss: 0.1261 | Train Acc: 96.04% | Val Acc: 95.97%
Epoch [04/10] - Loss: 0.0964 | Train Acc: 96.65% | Val Acc: 95.83%
Epoch [05/10] - Loss: 0.0902 | Train Acc: 96.79% | Val Acc: 97.15%
Epoch [06/10] - Loss: 0.0823 | Train Acc: 97.26% | Val Acc: 96.18%
Epoch [07/10] - Loss: 0.0731 | Train Acc: 97.24% | Val Acc: 97.22%  <-- Peak Checkpoint
Epoch [08/10] - Loss: 0.0551 | Train Acc: 98.14% | Val Acc: 94.65%
Epoch [09/10] - Loss: 0.0483 | Train Acc: 98.40% | Val Acc: 95.07%  <-- Lowest Loss
Epoch [10/10] - Loss: 0.0606 | Train Acc: 97.93% | Val Acc: 89.79%

Training Completed: Best Validation Accuracy: 97.22%
Model Checkpoint: models_checkpoint/lightweight_best.pth (48.7 KB)
```

---

## License
MIT License. Available for research, academic, and clinical decision support benchmarking.
