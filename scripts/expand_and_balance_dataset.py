"""
NeuroScan Dataset Expansion & Clinical Balancing Engine.
Integrates real multi-cohort public MRI datasets (Sartaj, Br35H, Navoneel, TCGA-LGG)
with clinical domain-specific data augmentations (Elastic deformation, Rician noise, Affine)
to produce a balanced 12,400+ scan benchmark across all 4 classes.
"""
import os
import glob
import hashlib
from typing import Dict, List, Set, Tuple
import numpy as np
import cv2

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TARGET_TRAIN_PER_CLASS = 2400
TARGET_TEST_PER_CLASS = 700
TARGET_TOTAL_PER_CLASS = TARGET_TRAIN_PER_CLASS + TARGET_TEST_PER_CLASS  # 3,100 per class = 12,400 total

def calculate_sha256(filepath: str) -> str:
    """Calculates SHA256 file hash to ensure zero duplicates."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(32768):
            h.update(chunk)
    return h.hexdigest()

def augment_elastic_fast(img_np: np.ndarray, alpha: float = 18.0, sigma: float = 4.0) -> np.ndarray:
    """High-speed vectorized elastic deformation via OpenCV remap."""
    h, w = img_np.shape[:2]
    # Downsample noise generation for speed, then upscale
    rand_x = (np.random.rand(h // 2, w // 2).astype(np.float32) * 2.0 - 1.0)
    rand_y = (np.random.rand(h // 2, w // 2).astype(np.float32) * 2.0 - 1.0)
    dx = cv2.resize(cv2.GaussianBlur(rand_x, (0, 0), sigma / 2.0) * alpha, (w, h))
    dy = cv2.resize(cv2.GaussianBlur(rand_y, (0, 0), sigma / 2.0) * alpha, (w, h))
    
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def augment_rician_and_gamma(img_np: np.ndarray) -> np.ndarray:
    """Simulates RF coil Rician noise and T1/T2 acquisition timing variance."""
    gamma = np.random.uniform(0.85, 1.15)
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted = cv2.LUT(img_np, lut)
    
    noise = np.random.normal(0, np.random.uniform(2.0, 5.0), img_np.shape)
    return np.clip(adjusted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def augment_affine(img_np: np.ndarray) -> np.ndarray:
    """Simulates head position shift and rotation within MRI scanner."""
    h, w = img_np.shape[:2]
    angle = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.96, 1.04)
    tx = np.random.uniform(-6, 6)
    ty = np.random.uniform(-6, 6)
    
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def run_expansion(dataset_root: str = "Dataset"):
    print("=" * 70)
    print("      NEUROSCAN CLINICAL DATASET EXPANSION & BALANCING ENGINE       ")
    print("=" * 70)
    
    # 1. Map existing files and hashes
    existing_hashes: Set[str] = set()
    counts: Dict[str, Dict[str, int]] = {"Training": {}, "Testing": {}}
    
    for split in ["Training", "Testing"]:
        for cls in CLASSES:
            folder = os.path.join(dataset_root, split, cls)
            os.makedirs(folder, exist_ok=True)
            files = [f for f in glob.glob(os.path.join(folder, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[split][cls] = len(files)
            for f in files:
                existing_hashes.add(calculate_sha256(f))

    print("\n[STEP 1] Current Dataset Status:")
    for cls in CLASSES:
        print(f"  - {cls.capitalize():12}: Training = {counts['Training'][cls]:4d} | Testing = {counts['Testing'][cls]:4d} | Total = {counts['Training'][cls] + counts['Testing'][cls]:4d}")
    print(f"  Total Verified Scans: {sum(counts['Training'].values()) + sum(counts['Testing'].values())}")
    print(f"  Unique SHA-256 Hashes: {len(existing_hashes)}")

    # 2. Ingest external real datasets
    print("\n[STEP 2] Ingesting New Scans from External Public Cohorts...")
    ingested_counts: Dict[str, int] = {c: 0 for c in CLASSES}

    sartaj_root = r"C:\Users\Administrator\.cache\kagglehub\datasets\sartajbhuvaji\brain-tumor-classification-mri\versions\3"
    br35h_root = r"C:\Users\Administrator\.cache\kagglehub\datasets\ahmedhamada0\brain-tumor-detection\versions\12"
    navoneel_root = r"C:\Users\Administrator\.cache\kagglehub\datasets\navoneel\brain-mri-images-for-brain-tumor-detection\versions\1"
    buda_root = r"C:\Users\Administrator\.cache\kagglehub\datasets\mateuszbuda\lgg-mri-segmentation\versions\2"

    candidate_sources: List[Tuple[str, str, str]] = []

    # Sartaj Ingestion
    if os.path.exists(sartaj_root):
        mapping = {
            "glioma_tumor": "glioma",
            "meningioma_tumor": "meningioma",
            "pituitary_tumor": "pituitary",
            "no_tumor": "notumor"
        }
        for sub, cls in mapping.items():
            for f in glob.glob(os.path.join(sartaj_root, "**", sub, "*.*"), recursive=True):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    candidate_sources.append((f, cls, "sartaj"))

    # Br35H Ingestion
    if os.path.exists(br35h_root):
        for f in glob.glob(os.path.join(br35h_root, "no", "*.*")):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                candidate_sources.append((f, "notumor", "br35h"))

    # Navoneel Ingestion
    if os.path.exists(navoneel_root):
        for f in glob.glob(os.path.join(navoneel_root, "no", "*.*")):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                candidate_sources.append((f, "notumor", "navoneel"))

    # TCGA-LGG Ingestion (Glioma)
    if os.path.exists(buda_root):
        for f in glob.glob(os.path.join(buda_root, "**", "*.tif"), recursive=True):
            if not f.endswith("_mask.tif"):
                candidate_sources.append((f, "glioma", "tcga_lgg"))

    print(f"  Evaluating {len(candidate_sources)} external candidates for deduplication...")

    for src_path, target_cls, src_tag in candidate_sources:
        try:
            curr_train = counts["Training"][target_cls]
            curr_test = counts["Testing"][target_cls]

            if curr_train >= TARGET_TRAIN_PER_CLASS and curr_test >= TARGET_TEST_PER_CLASS:
                continue

            h = calculate_sha256(src_path)
            if h in existing_hashes:
                continue
            existing_hashes.add(h)

            if curr_train < TARGET_TRAIN_PER_CLASS:
                dest_dir = os.path.join(dataset_root, "Training", target_cls)
                counts["Training"][target_cls] += 1
            else:
                dest_dir = os.path.join(dataset_root, "Testing", target_cls)
                counts["Testing"][target_cls] += 1

            fname = f"real_{src_tag}_{h[:8]}.jpg"
            dest_file = os.path.join(dest_dir, fname)

            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dest_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                ingested_counts[target_cls] += 1
        except Exception:
            continue

    for cls in CLASSES:
        print(f"  - Ingested new real scans for [{cls}]: +{ingested_counts[cls]} scans")

    # 3. Clinical Data Augmentation for Remaining Balance
    print("\n[STEP 3] Applying Clinical Data Augmentations (Elastic / Rician / Affine)...")
    for split in ["Training", "Testing"]:
        target_quota = TARGET_TRAIN_PER_CLASS if split == "Training" else TARGET_TEST_PER_CLASS
        for cls in CLASSES:
            folder = os.path.join(dataset_root, split, cls)
            current_files = [f for f in glob.glob(os.path.join(folder, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            needed = target_quota - len(current_files)
            
            if needed <= 0:
                continue
            
            print(f"  Generating {needed} clinical augmentations for [{split}/{cls}]...")
            idx = 0
            while needed > 0 and len(current_files) > 0:
                base_file = current_files[idx % len(current_files)]
                idx += 1
                try:
                    img_np = cv2.imread(base_file)
                    if img_np is None:
                        continue
                    
                    aug_type = idx % 3
                    if aug_type == 0:
                        aug_np = augment_elastic_fast(img_np)
                        tag = "elastic"
                    elif aug_type == 1:
                        aug_np = augment_rician_and_gamma(img_np)
                        tag = "rician"
                    else:
                        aug_np = augment_affine(img_np)
                        tag = "affine"
                    
                    aug_name = f"aug_{tag}_{cls}_{split.lower()}_{idx:04d}.jpg"
                    aug_path = os.path.join(folder, aug_name)
                    cv2.imwrite(aug_path, aug_np, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                    
                    needed -= 1
                    counts[split][cls] += 1
                except Exception:
                    continue

    # 4. Final Summary Table
    print("\n" + "=" * 70)
    print("                 FINAL DATASET BENCHMARK METRICS                   ")
    print("=" * 70)
    final_counts: Dict[str, Dict[str, int]] = {"Training": {}, "Testing": {}}
    for split in ["Training", "Testing"]:
        for cls in CLASSES:
            folder = os.path.join(dataset_root, split, cls)
            final_counts[split][cls] = len([f for f in glob.glob(os.path.join(folder, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"{'Class':<15} | {'Training':<10} | {'Testing':<10} | {'Total':<10} | {'Status':<12}")
    print("-" * 65)
    for cls in CLASSES:
        tr = final_counts["Training"][cls]
        te = final_counts["Testing"][cls]
        tot = tr + te
        status = "BALANCED" if (tr >= TARGET_TRAIN_PER_CLASS and te >= TARGET_TEST_PER_CLASS) else "EXPANDED"
        print(f"{cls.capitalize():<15} | {tr:<10d} | {te:<10d} | {tot:<10d} | {status:<12}")

    total_train = sum(final_counts["Training"].values())
    total_test = sum(final_counts["Testing"].values())
    print("-" * 65)
    print(f"{'GRAND TOTAL':<15} | {total_train:<10d} | {total_test:<10d} | {total_train + total_test:<10d} | READY")
    print("=" * 70)

if __name__ == "__main__":
    run_expansion()
