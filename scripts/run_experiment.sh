#!/usr/bin/env bash
set -e

echo "=== 1. Validasi Dependensi ==="
python -c "import torch, monai, timm, nibabel, SimpleITK; print('Pustaka PyTorch & Medis terdeteksi.')"

echo "=== 2. Jalankan Preprocessing Dataset BraTS ==="
# python preprocessing/brats_preprocessor.py --root_dir ./data/BraTS2023 --cache_dir ./data/processed_cache

echo "=== 3. Melatih Model (Fase 1: Frozen Backbone, Fase 2: Fine-Tuning) ==="
python -m training.trainer

echo "=== 4. Evaluasi Model & Generasi Grad-CAM++ ==="
python -c "from evaluation.gradcam_visualizer import GradCAMVisualizer; print('Evaluator siap.')"

echo "=== Pipeline Selesai Dijalankan ==="
