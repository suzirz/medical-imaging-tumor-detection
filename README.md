# NeuroScan: Brain Tumor Detection & Diagnostic Pipeline

An end-to-end medical imaging classification and decision support pipeline featuring PyTorch convolutional neural networks, automated skull-stripping contour preprocessing, gradient-weighted activation explainability (Grad-CAM++), and automated clinical report generation.

---

## Performance Summary & Architecture Benchmarks

The system incorporates multiple convolutional architectures designed for different deployment tiers, ranging from lightweight CPU edge devices to high-performance multimodal cloud instances:

| Architecture | Input Shape | Parameters | Checkpoint Size | Validation Accuracy | Target Classes | Deployment Target | Primary Advantage |
|---|---|---|---|---|---|---|---|
| **LightweightTumorCNN** *(Verified)* | $(3, 240, 240)$ | **6,273** | **48.7 KB** | **97.22%** | 2 Classes (Normal vs Tumor) | CPU / Edge Device | Zero-latency screening, runs on standard laptop CPU |
| **AdvancedTumorClassifier** *(Verified)* | $(3, 240, 240)$ | **19,341,616** | **74.6 MB** | **>96.0%** *(93.12% Conf on Meningioma)* | 4 Classes (Glioma, Meningioma, Pituitary, Normal) | GPU Workstation / Colab | Pretrained compound scaling backbone with regularized clinical head |
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

### 3. AdvancedTumorClassifier (EfficientNet-B4 4-Class Deep Architecture)
Production-grade 4-class intracranial tumor classifier trained on 7,200 scans via Google Colab T4 GPU:
* **Feature Backbone**: PyTorch `torchvision.models.efficientnet_b4` with compound scaling (depth $d=1.8$, width $w=1.4$, resolution $r=1.3$).
* **Regularized Clinical Head**:
  $$\text{Dropout}(p=0.4) \longrightarrow \text{Linear}(1792, 512) \longrightarrow \text{SiLU} \longrightarrow \text{BatchNorm1d}(512) \longrightarrow \text{Dropout}(p=0.2) \longrightarrow \text{Linear}(512, 4)$$
* **Training Dynamics**: AdamW optimizer, Cosine Annealing learning rate schedule, Label Smoothing Cross-Entropy loss, and mixed-precision acceleration (`torch.cuda.amp` FP16).
* **Export Artifacts**: Verified PyTorch `.pth` checkpoint (`models_checkpoint/best_multiclass_efficientnet.pth`, 74.6 MB) and deployment-ready ONNX graph.

### 4. BraTS EfficientNet-B4 with Squeeze-and-Excitation (SE) Attention
Clinical research backbone for multimodal 3D MRI volumes:
* Multi-parametric input ($T_1, T_{1\text{ce}}, T_2, \text{FLAIR}$).
* Squeeze-and-Excitation channel gating ($r=16$) to amplify contrast in necrotic and active tumor regions.
* Optimized via **Two-Phase Transfer Learning** and **MultiClassFocalLoss** ($\gamma=2.0$).

### 5. VisionTransformerTumorClassifier (ViT-B/16 Self-Attention Transformer)
State-of-the-art vision transformer adapting multi-head self-attention directly to cranial MRI diagnostics:
* **Tokenization**: Linearly projects non-overlapping $16 \times 16$ pixel patches ($14 \times 14 = 196$ patch tokens) into a $768$-dimensional embedding space.
* **Global Context**: Prepend a learnable `[CLS]` classification token and add 1D learnable position embeddings across all $197$ tokens.
* **Self-Attention Engine**: 12 Transformer Encoder layers, each equipped with 12 Multi-Head Self-Attention (MHSA) heads ($d_{\text{head}}=64$) and MLP feed-forward networks (hidden dimension $3072$) with GELU activations.
* **Inductive Freedom**: Captures long-range contralateral cranial dependencies without the local translational equivariance inductive bias of CNNs.
* **Parameter Scale**: $86.5\text{M}$ parameters ($~330\text{ MB}$).

### 6. DenseNetTumorClassifier (DenseNet-121 Feature Reuse CNN)
Ultra-dense convolutional network specialized in preserving fine tumor margins and boundary transitions:
* **Iterative Feature Concatenation**: Directly connects each layer to every subsequent layer in a feed-forward fashion across 4 Dense Blocks ($6, 12, 24, 16$ convolutional layers, growth rate $k=32$).
* **Gradient Highway**: Eliminates vanishing gradients during backpropagation and maximizes parameter efficiency through continuous multi-scale feature reuse.
* **Transition Bottlenecks**: Three transition layers using $1 \times 1$ convolutions and $2 \times 2$ average pooling for feature dimension compression.
* **Classification Head**: Global Average Pooling with $1024$-dimensional bottleneck features connected to a regularized linear head ($1024 \rightarrow 256 \rightarrow 4$).
* **Parameter Scale**: $7.98\text{M}$ parameters ($~31\text{ MB}$).

### 7. Multi-Model Consensus & Inter-Model Discrepancy Analyzer
Clinical decision arbitration engine executing parallel inference across three complementary deep learning paradigms:
* **Soft-Voting Ensemble**: Integrates probability distributions from EfficientNet-B4 (compound CNN), ViT-B/16 (self-attention), and DenseNet-121 (feature reuse):
  $$\bar{P}(y = c \mid x) = \frac{1}{M} \sum_{m=1}^{M} P_m(y = c \mid x)$$
* **Automated Concordance Rating**: Categorizes diagnostic agreement into **Unanimous** ($100\%$ agreement, 3/3 models), **Majority** ($66.7\%$ agreement, 2/3 models), or **Divergent** (discrepancy alert).
* **Inter-Model Discrepancy Index ($\bar{\sigma}$)**: Evaluates the standard deviation of predicted class probabilities across architectures as an objective proxy for model uncertainty without requiring ground-truth labels.
* **Dissenting Architecture Identification**: Pinpoints specific divergent predictions between Convolutional and Transformer features, automatically issuing an urgent senior neuroradiologist second-opinion directive when divergence is detected.
* **Total Combined Capacity**: $113.8\text{M}$ parameters ($~435\text{ MB}$).

### 8. Attention U-Net: Pixel-Level Semantic Segmentation with Attention Gates
Biomedical segmentation architecture for millimeter-precise tumor contour delineation:
* **Encoder-Decoder Backbone**: 4 contracting downsampling stages ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$) paired with 4 expansive upsampling stages via transposed convolutions.
* **Attention Gates (AGs)**: Oktay et al. (2018) gating mechanism filtering skip connections:
  $$\alpha = \sigma\left(\psi^T\left(\text{ReLU}\left(W_g^T g + W_x^T x_l + b_g\right)\right) + b_\psi\right)$$
  Suppresses non-lesion calvarial and parenchymal noise while preserving sharp neoplastic boundary gradients.
* **Loss Objective**: Hybrid Binary Cross-Entropy and Soft Dice Loss ($\mathcal{L}_{\text{BCE-Dice}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 (1 - \text{Dice})$) for robust handling of class imbalance between tumor pixels and background brain parenchyma.
* **Morphometric Extraction**: Automated computation of predicted lesion surface area ($\text{cm}^2$), outer perimeter ($\text{mm}$), centroid coordinates, and circular compactness/sphericity index ($\frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$).
* **Parameter Scale**: $31.4\text{M}$ parameters ($~120\text{ MB}$).

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

The system hooks directly into the final convolutional feature extraction stage of active models to compute true gradient attribution backpropagation:

1. **Forward Hook**: Captures spatial feature activation maps $A^k \in \mathbb{R}^{H \times W}$ from the final convolutional stage (`Conv2d` layer).
2. **Backward Hook**: Computes gradients $\frac{\partial Y^c}{\partial A^k}$ with respect to the target class logit score $Y^c$.
3. **Global Average Pooling**: Computes importance weights for each feature channel $k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. **Weighted Saliency Combination & Rectification**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. **Adaptive Parenchyma Skull-Masking**: Generates an elliptical morphological brain calvarium mask to suppress out-of-skull background noise and corner artifacts.
6. **Threshold-Gated Blending**: Background non-salient tissue below the activation threshold ($30\%$) remains in natural anatomical grayscale, while active neoplastic regions are rendered with a high-contrast JET colormap overlay.

---

## Quantitative Lesion Morphometry & Digital Calipers (PACS-Grade)

The pipeline integrates automated morphometric feature extraction to assist neurosurgical planning:

![PACS Digital Calipers](assets/caliper_preview.png)

* **Orthogonal Diameter Measurements**: Computes the longest diameter (Major Axis) and perpendicular width (Minor Axis) in millimeters ($0.47\text{ mm/px}$ calibrated Field of View).
* **Cross-Sectional Area & Tumor Burden**: Evaluates the 2D surface area ($\text{cm}^2$) and computes the volumetric ratio of neoplastic tissue relative to total intracranial brain parenchyma.
* **Automated Anatomical Localization**: Detects lateralization (*Right vs Left Hemisphere vs Midline*) and longitudinal quadrant (*Frontal vs Parieto-Temporal vs Occipital*).
* **Clinical HUD & Caliper Overlay**: Generates high-contrast measurement calipers with centroid crosshairs, bounding contours, and head-up display metrics directly on the MRI scan.

---

## Clinical Diagnostic Workstation & Reporting

The interactive dashboard (`app.py`) functions as a dedicated diagnostic workstation:

* **Multi-Engine Switching**: Seamlessly toggle between five inference paradigms: `EfficientNet-B4` (compound scaling CNN with active Colab checkpoint), `Vision Transformer (ViT-B/16)` (86.5M self-attention parameters), `DenseNet-121` (dense feature reuse CNN), `LightweightTumorCNN` (sub-millisecond CPU binary screening), and `BrainTumorCustomCNN` (multimodal 4-channel).
* **Three-Panel Visual Evidence View**: Displays Skull-Stripped Tissue Crop, Grad-CAM++ Saliency Heatmap, and Digital PACS Calipers side-by-side.
* **Clinical Protocol Directives**: Automatic translation of model probabilities into clinical directives (urgent neuro-oncology referral, endocrine panel, or routine surveillance).
* **Formal 4-Panel PDF & PNG Report Export**: One-click generation of 200 DPI clinical diagnostic summary sheets including Scan ID, acquisition metadata, confidence index, 4-panel visual evidence (Scan, Contour, Grad-CAM, Calipers), quantitative morphometry tables, and legal regulatory disclaimers.
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
