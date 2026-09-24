# Medical Imaging Tumor Detection Pipeline: BraTS 2023 + EfficientNet-B4

Pipeline klasifikasi dan deteksi tumor otak berbasis PyTorch, MONAI, EfficientNet-B4, FastAPI, dan Grad-CAM++. Diadaptasi dari spesifikasi arsitektur AWS PartyRock.

---

## Spesifikasi Pipeline

| Komponen | Spesifikasi Teknis |
|---|---|
| Modalitas Input | 4-Channel MRI Multimodal: T1, T1ce, T2, FLAIR (.nii.gz) |
| Target Klasifikasi | 4 Kelas: Normal (0), Glioma (1), Meningioma (2), Tumor Hipofisis (3) |
| Backbone Model | EfficientNet-B4 (input 4-channel) + Squeeze-and-Excitation Attention |
| Fungsi Loss | MultiClassFocalLoss ($\gamma=2.0, \alpha=[0.25, 0.25, 0.25, 0.25]$) |
| Pelatihan | Transfer Learning 2-Fase: Fase 1 (Head Only), Fase 2 (Full Fine-Tuning) |
| Explainability (XAI) | Grad-CAM++ dan pemetaan rekomendasi klinis otomatis |
| Deployment | REST API FastAPI, Docker multi-stage, docker-compose GPU |

---

## Struktur Direktori

```text
├── api/
│   └── main.py                             # Endpoint REST API (/predict, /health)
├── docs/
│   └── troubleshooting_and_optimization.md # Catatan teknis & penanganan error
├── models/
│   └── efficientnet_tumor_classifier.py    # EfficientNet-B4 4-ch, SE Block, Focal Loss
├── preprocessing/
│   └── brats_preprocessor.py               # Loader NIfTI, Z-score, N4 bias, ekstraksi aksial
├── training/
│   └── trainer.py                          # Training loop 2-fase PyTorch
├── evaluation/
│   └── gradcam_visualizer.py               # Visualisasi Grad-CAM++ & action map klinis
├── app.py                                  # Web demo interaktif (Streamlit)
├── requirements.txt                        # Versi pustaka teruji
├── Dockerfile                              # Build container multi-stage
├── docker-compose.yml                      # Orkestrasi container dengan GPU passthrough
└── CHAT.md                                 # Log dan konteks proyek
```

---

## Panduan Menjalankan

### 1. Pasang Dependensi
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Latih Model (Dua Fase)
```bash
python -m training.trainer
```

### 3. Jalankan REST API FastAPI
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Akses dokumentasi Swagger interaktif di: `http://localhost:8000/docs`.

### 4. Jalankan Web Demo (Streamlit)
```bash
streamlit run app.py
```

### 5. Jalankan dengan Docker
```bash
docker-compose up --build
```
