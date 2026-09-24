# Brain Tumor Detection Pipeline (BraTS 2023 & EfficientNet-B4)

End-to-end multimodal brain tumor classification and detection pipeline using PyTorch, MONAI, EfficientNet-B4, and Grad-CAM++ explainability.

---

## About the Project

Medical image analysis for brain tumors requires high precision across diverse scan types. While conventional projects focus on simple binary classification on 2D images, this repository implements a clinical-grade pipeline supporting multimodal 3D MRI scans (BraTS 2023) across four diagnostic classes:
* **Normal** (No tumor detected)
* **Glioma**
* **Meningioma**
* **Pituitary Tumor**

---

## Dataset and Preprocessing

### Data Modalities
The pipeline processes multimodal MRI volumes containing four co-registered sequences:
1. **T1**: T1-weighted
2. **T1ce**: T1-weighted contrast enhanced
3. **T2**: T2-weighted
4. **FLAIR**: Fluid-attenuated inversion recovery

### Preprocessing Pipeline
Each volume undergoes standardized medical imaging preprocessing:
1. **Bias Field Correction**: N4ITK via SimpleITK to eliminate RF field inhomogeneities.
2. **Skull Stripping**: Removes non-brain intracranial tissue and cranial bone artifacts.
3. **Intensity Normalization**: Z-score normalization computed exclusively over non-zero brain voxels.
4. **Isotropic Resampling**: Resamples spatial voxel spacing to $1.0 \times 1.0 \times 1.0\text{ mm}$ using BSpline interpolation.
5. **Axial Slice Extraction**: Extracts the middle 60% of axial slices, discarding top and bottom 20% peripheral noise, stacked into a 4-channel tensor $(4, 380, 380)$.

---

## Model Architecture

```text
Input (4, 380, 380)
       │
       ▼
EfficientNet-B4 Backbone (Pretrained, modified 4-channel first conv)
       │
       ▼
Squeeze-and-Excitation (SE) Channel Attention (Reduction Ratio = 16)
       │
       ▼
Adaptive Average Pooling + Flatten (1792 features)
       │
       ▼
BatchNorm1d -> Linear(1792, 512) -> SiLU -> Dropout(0.4)
       │
       ▼
Linear(512, 256) -> SiLU -> Dropout(0.2)
       │
       ▼
Linear(256, 4) -> Softmax (Normal, Glioma, Meningioma, Pituitary)
```

### Why this Architecture?
* **4-Channel Input**: Combining T1, T1ce, T2, and FLAIR allows the network to evaluate tissue contrast simultaneously.
* **SE Attention**: Adaptively re-weights channel importance to highlight tumor-rich contrasts while suppressing background signals.
* **Multi-Class Focal Loss**: Handles severe class imbalance ($\gamma=2.0, \alpha=[0.25, 0.25, 0.25, 0.25]$) by focusing gradient updates on hard examples.

---

## Training Strategy

Training uses a two-phase transfer learning approach:

| Phase | Description | Epochs | Optimizer & LR | Scheduler |
|---|---|---|---|---|
| **Phase 1** | Freeze backbone; train classification head only | 20 | AdamW, $\text{LR}=10^{-3}$ | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Phase 2** | Unfreeze all layers; fine-tune entire network | 80 | AdamW, $\text{LR}=10^{-5}$ | CosineAnnealingLR ($T_{\max}=80, \eta_{\min}=10^{-7}$) |

Mixed precision (AMP) and gradient clipping ($\text{max\_norm}=1.0$) are enforced throughout training.

---

## Evaluation and Explainability (Grad-CAM++)

The model integrates **Grad-CAM++** to generate spatial activation heatmaps overlaid on the original MRI scans.

### Automated Clinical Decision Mapping
* **Normal**: Routine follow-up; no immediate surgical intervention required.
* **Glioma**: Urgent neurosurgical oncology referral.
* **Meningioma**: Neurology consultation and mass progression monitoring.
* **Pituitary Tumor**: Endocrinology evaluation and surgical consultation.
* **Low Confidence ($< 0.50$)**: Flagged as *Uncertain/Borderline* for manual radiologist audit.

---

## Repository Structure

```text
├── api/
│   └── main.py                             # FastAPI REST endpoints (/predict, /health)
├── config/
│   └── default_config.json                 # Training & pipeline configuration
├── evaluation/
│   └── gradcam_visualizer.py               # Grad-CAM++ heatmaps & clinical action mapping
├── models/
│   ├── efficientnet_tumor_classifier.py    # EfficientNet-B4 4-ch, SE Block, Focal Loss
│   └── predictor.py                        # Standalone inference engine (Deep Module)
├── notebooks/
│   └── brats_efficientnet_colab.ipynb      # 10-step Google Colab execution notebook
├── preprocessing/
│   └── brats_preprocessor.py               # NIfTI loader, Z-score, N4 bias, axial extraction
├── scripts/
│   └── run_experiment.sh                   # Pipeline execution script
├── training/
│   └── trainer.py                          # Two-phase PyTorch training loop
├── app.py                                  # Interactive Streamlit web interface
├── requirements.txt                        # Pinned dependencies
├── Dockerfile                              # Multi-stage production container
└── docker-compose.yml                      # GPU-enabled service orchestration
```

---

## Quick Start

### 1. Installation
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Run locally (CPU / CUDA GPU):
python -m training.trainer

# Or execute the step-by-step notebook on Google Colab:
# Open notebooks/brats_efficientnet_colab.ipynb in Colab with T4 GPU runtime.
```

### 3. Run FastAPI Production Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation: `http://localhost:8000/docs`.

### 4. Interactive Web Interface (Streamlit)
```bash
streamlit run app.py
```

### 5. Production Docker Deployment
```bash
docker-compose up --build
```

---

## License
MIT License.
