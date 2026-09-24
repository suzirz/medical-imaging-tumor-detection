# Medical Imaging Tumor Detection Pipeline: BraTS 2023 + EfficientNet-B4

An end-to-end multimodal brain tumor classification and detection pipeline built with PyTorch, MONAI, EfficientNet-B4, FastAPI, and Grad-CAM++. Adapted from the AWS PartyRock medical imaging pipeline design.

---

## Technical Specifications

| Component | Specification |
|---|---|
| Input Modalities | 4-Channel Multimodal MRI: T1, T1ce, T2, FLAIR (.nii.gz) |
| Classification Targets | 4 Classes: Normal (0), Glioma (1), Meningioma (2), Pituitary Tumor (3) |
| Model Backbone | EfficientNet-B4 (modified 4-channel input) + Squeeze-and-Excitation Attention |
| Loss Function | MultiClassFocalLoss ($\gamma=2.0, \alpha=[0.25, 0.25, 0.25, 0.25]$) |
| Training Strategy | Two-Phase Transfer Learning: Phase 1 (Frozen Backbone), Phase 2 (Full Fine-Tuning) |
| Explainability (XAI) | Grad-CAM++ with automated clinical recommendation mapping |
| Deployment | FastAPI REST API, multi-stage Docker build, GPU-enabled docker-compose |

---

## Repository Structure

```text
├── api/
│   └── main.py                             # FastAPI REST endpoints (/predict, /health)
├── config/
│   └── default_config.json                 # Pipeline & training hyperparameters
├── evaluation/
│   └── gradcam_visualizer.py               # Grad-CAM++ activation maps & clinical actions
├── models/
│   ├── efficientnet_tumor_classifier.py    # EfficientNet-B4 4-ch, SE Block, Focal Loss
│   └── predictor.py                        # Self-contained inference engine (Deep Module)
├── notebooks/
│   └── brats_efficientnet_colab.ipynb      # 10-step Google Colab execution notebook
├── preprocessing/
│   └── brats_preprocessor.py               # NIfTI loader, Z-score, N4 bias, axial extraction
├── scripts/
│   └── run_experiment.sh                   # End-to-end pipeline execution script
├── training/
│   └── trainer.py                          # Two-phase PyTorch training loop
├── app.py                                  # Interactive Streamlit web interface
├── requirements.txt                        # Pinned dependencies
├── Dockerfile                              # Multi-stage container build
├── docker-compose.yml                      # Service orchestration with NVIDIA GPU support
└── setup.py                                # Package installation configuration
```

---

## Getting Started

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the Model (Two-Phase Strategy)
```bash
python -m training.trainer
```
Alternatively, open and run `notebooks/brats_efficientnet_colab.ipynb` on Google Colab with free GPU acceleration.

### 3. Run the FastAPI REST Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation will be available at `http://localhost:8000/docs`.

### 4. Launch the Streamlit Web Interface
```bash
streamlit run app.py
```

### 5. Deploy with Docker
```bash
docker-compose up --build
```

---

## License
MIT License.
