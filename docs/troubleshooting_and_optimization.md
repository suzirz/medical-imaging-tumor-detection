# 🧠 Panduan Troubleshooting, Optimasi, dan Best Practices Medis

Panduan teknis ini dirancang khusus untuk pipeline **Deteksi Tumor Otak BraTS 2023 + EfficientNet-B4 + MONAI**.

---

## 1. ⚠️ Kesalahan Umum & Solusinya

### A. Format Volume NIfTI Tidak Cocok / Dimensi Salah
* **Pola Error**: `ValueError: could not broadcast input array from shape (240, 240, 155) into shape (240, 240, 150)`
* **Penyebab**: Modalitas scan T1, T2, atau FLAIR diambil dengan slice thickness/FOV berbeda.
* **Solusi**: Gunakan SimpleITK `ResampleImageFilter` untuk menyamakan isotropic spacing menjadi `(1.0, 1.0, 1.0) mm` sebelum ekstraksi slice:
```python
resample = sitk.ResampleImageFilter()
resample.SetInterpolator(sitk.sitkBSpline)
resample.SetOutputSpacing([1.0, 1.0, 1.0])
```

### B. CUDA Out of Memory (OOM) Selama Pelatihan
* **Pola Error**: `RuntimeError: CUDA out of memory. Tried to allocate...`
* **Solusi**:
  1. Aktifkan PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`).
  2. Turunkan batch size menjadi `8` atau `16`, dan gunakan **Gradient Accumulation** (misal akumulasi 2 langkah):
```python
scaler.scale(loss / 2).backward()
if (step + 1) % 2 == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### C. Numerical Instability pada Focal Loss (Loss Bernilai NaN / Inf)
* **Penyebab**: Nilai logaritma mendekati 0 saat prediksi probabilitas sangat yakin.
* **Solusi**: Gunakan `torch.clamp` dengan epsilon batas bawah `1e-7` pada `MultiClassFocalLoss`.

---

## 2. ⚡ Strategi Optimasi Performa GPU

| Spesifikasi GPU | Batch Size Optimal | Mixed Precision (AMP) | Rekomendasi Worker |
|---|---|---|---|
| **8 GB VRAM** (RTX 3070 / 4060) | 8 | Wajib (FP16) | `num_workers=2` |
| **16 GB VRAM** (T4 / V100 16G) | 16 | Wajib (FP16) | `num_workers=4` |
| **24 GB VRAM** (RTX 3090 / 4090) | 32 | Wajib (FP16/BF16) | `num_workers=8` |
| **40-80 GB VRAM** (A100 / H100) | 64 | Wajib (BF16) | `num_workers=12` |

---

## 3. 🎯 Strategi Ketidakseimbangan Kelas (Class Imbalance)
Dataset BraTS memiliki rasio kasus tumor yang tidak seragam (Glioma lebih mendominasi daripada Meningioma atau Normal).
1. **WeightedRandomSampler**: Mengatur frekuensi sampling per batch agar kelas minoritas dipelajari secara proporsional.
2. **Focal Loss**: Memberikan bobot $\gamma=2.0$ untuk meredam penalti dari sampel yang mudah diklasifikasikan (*easy negatives*).

---

## 4. 🏥 Pertimbangan Klinis & Explainable AI (Grad-CAM++)
* **Threshold Kepercayaan**: Bila model memiliki confidence $< 0.50$, sistem **wajib** mengembalikan status `Hasil Tidak Pasti (Borderline)` dan menandai rujukan peninjauan manual oleh Dokter Spesialis Radiologi.
* **Validasi Peta Panas**: Pastikan fokus aktivasi Grad-CAM++ berada tepat di dalam struktur jaringan intrakranial, bukan artefak tengkorak (*skull bone*).
