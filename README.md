# 🩺 Medical Imaging Tumor Detection Pipeline: BraTS 2023 + EfficientNet-B4

Sistem klasifikasi dan deteksi tumor otak multimodal berbasis Deep Learning (**PyTorch + MONAI + EfficientNet-B4 + FastAPI + Grad-CAM++**), diadaptasi dari arsitektur rancangan AWS PartyRock.

---

## 🏗️ Arsitektur & Spesifikasi Pipeline

| Komponen | Spesifikasi Teknis |
|---|---|
| **Modalitas Input** | 4-Channel MRI Multimodal: `T1`, `T1ce`, `T2`, `FLAIR` (.nii.gz) |
| **Target Klasifikasi** | 4 Kelas: `0: Normal` · `1: Glioma` · `2: Meningioma` · `3: Tumor Hipofisis` |
| **Backbone Model** | `EfficientNet-B4` (modifikasi input 4-channel) + Squeeze-and-Excitation (SE) Attention |
| **Loss Function** | `MultiClassFocalLoss` ($\gamma=2.0, \alpha=[0.25, 0.25, 0.25, 0.25]$) |
| **Strategi Pelatihan** | Transfer Learning 2-Fase (Fase 1: Frozen Backbone $\rightarrow$ Fase 2: Full Fine-Tuning) |
| **Explainable AI (XAI)** | `Grad-CAM++` dengan pemetaan rekomendasi klinis otomatis |
| **Deployment** | Backend REST API `FastAPI`, `Docker`, dan `docker-compose` (GPU passthrough) |

---

## 📁 Struktur Direktori

```text
├── api/
│   └── main.py                     # Layanan REST API FastAPI (/predict, /health)
├── docs/
│   └── troubleshooting_and_optimization.md # Panduan troubleshooting & optimasi GPU
├── models/
│   └── efficientnet_tumor_classifier.py    # EfficientNet-B4 4-ch + SE Block + Focal Loss
├── preprocessing/
│   └── brats_preprocessor.py       # Pemuatan NIfTI, Z-score, N4 bias, slice 60% aksial
├── training/
│   └── trainer.py                  # Loop pelatihan 2-fase PyTorch (AdamW + Cosine Annealing)
├── evaluation/
│   └── gradcam_visualizer.py       # Generator Heatmap Grad-CAM++ & Action Map Klinis
├── app.py                          # UI Web Interaktif (Streamlit)
├── requirements.txt                # Dependensi versi terpinned (PyTorch, MONAI, timm, dll)
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # GPU-enabled container orchestration
└── CHAT.md                         # Memori riwayat proyek
```

---

## 🚀 Panduan Menjalankan

### 1. Instalasi Dependensi
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Melatih Model (Dua Fase)
```bash
python -m training.trainer
```

### 3. Menjalankan REST API FastAPI (Produksi)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Buka dokumentasi Swagger interaktif di: `http://localhost:8000/docs`.

### 4. Menjalankan Antarmuka Web Interaktif (Streamlit)
```bash
streamlit run app.py
```

### 5. Deployment Menggunakan Docker
```bash
docker-compose up --build
```
