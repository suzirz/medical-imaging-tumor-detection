# Panduan Troubleshooting dan Optimasi

Catatan teknis penanganan kendala dan optimasi pipeline klasifikasi tumor otak BraTS 2023 dengan EfficientNet-B4 dan MONAI.

---

## 1. Kendala Umum dan Solusi

### A. Dimensi Volume NIfTI Tidak Seragam
* **Pola Masalah**: Volume scan antar modalitas (T1, T2, FLAIR) memiliki resolusi spasial atau slice thickness berbeda.
* **Solusi**: Samakan isotropic spacing menjadi `(1.0, 1.0, 1.0) mm` memakai `sitk.ResampleImageFilter` sebelum ekstraksi slice:
```python
resample = sitk.ResampleImageFilter()
resample.SetInterpolator(sitk.sitkBSpline)
resample.SetOutputSpacing([1.0, 1.0, 1.0])
```

### B. Kehabisan Memori GPU (CUDA OOM)
* **Solusi**:
  1. Gunakan PyTorch Mixed Precision (`torch.cuda.amp.autocast`).
  2. Set batch size ke `8` atau `16`, lalu terapkan akumulasi gradien jika butuh batch efektif lebih besar:
```python
scaler.scale(loss / 2).backward()
if (step + 1) % 2 == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### C. Stabilitas Numerik Focal Loss (NaN atau Inf)
* **Penyebab**: Logaritma mendekati nol saat prediksi probabilitas sangat tinggi.
* **Solusi**: Batasi probabilitas menggunakan clamp dengan nilai batas bawah `1e-7`.

---

## 2. Kapasitas GPU dan Ukuran Batch

| VRAM GPU | Batch Size | Mixed Precision | Worker DataLoader |
|---|---|---|---|
| 8 GB | 8 | Aktif (FP16) | 2 |
| 16 GB | 16 | Aktif (FP16) | 4 |
| 24 GB | 32 | Aktif (FP16/BF16) | 8 |
| 40-80 GB | 64 | Aktif (BF16) | 12 |

---

## 3. Penanganan Ketidakseimbangan Kelas
Jumlah sampel tiap jenis tumor pada dataset BraTS tidak sama rata.
1. **WeightedRandomSampler**: Menyamakan frekuensi sampling tiap batch agar kelas minoritas mendapat porsi seimbang selama iterasi.
2. **Focal Loss**: Meredam penalti dari sampel yang mudah diklasifikasikan dengan parameter $\gamma=2.0$.

---

## 4. Validasi Klinis dan Explainability
* **Ambang Kepercayaan**: Jika confidence score model berada di bawah `0.50`, output dikelompokkan sebagai status belum pasti dan diarahkan untuk peninjauan manual radiolog.
* **Inspeksi Grad-CAM++**: Verifikasi bahwa aktivasi peta panas berada pada parenkim otak, bukan pada area artefak tengkorak.
