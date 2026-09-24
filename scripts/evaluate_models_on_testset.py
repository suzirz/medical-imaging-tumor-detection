"""
NeuroScan Empirical Test Set Evaluator (Pure NumPy Implementation).
Executes rigorous, verifiable evaluation of trained checkpoints across the
official 2,800 held-out test cohort in Dataset/Testing/.
Computes real Confusion Matrix, Precision, Recall, Specificity, F1, and Latency metrics.
Zero external scientific dependencies (pure NumPy & PyTorch).
"""
import os
import sys
import time
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.advanced_classifier import AdvancedTumorClassifier
from models.lightweight_cnn import LightweightTumorCNN

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = ["Glioma", "Meningioma", "Normal (No Tumor)", "Pituitary Adenoma"]

def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(int(cm[c, :].sum()))
        
    return cm, np.array(precisions), np.array(recalls), np.array(f1s), np.array(supports)

def evaluate_efficientnet(test_dir="Dataset/Testing", ckpt_path="models_checkpoint/best_multiclass_efficientnet.pth"):
    print("\n" + "="*70)
    print(" 1. EMPIRICAL EVALUATION: EfficientNet-B4 (4-Class Multi-Category)")
    print("="*70)
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path}")
        return None

    device = torch.device("cpu")
    model = AdvancedTumorClassifier(num_classes=4, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_transforms = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_dir, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Dataset classes mapped: {test_dataset.class_to_idx}")
    print(f"Total test cohort: {len(test_dataset)} scans across {len(test_dataset.classes)} balanced classes")

    y_true = []
    y_pred = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.time()
            outputs = model(images)
            latencies.append((time.time() - t0) / len(images))
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_acc = (y_true == y_pred).mean() * 100.0
    avg_latency_ms = np.mean(latencies) * 1000.0

    cm, prec, rec, f1, supp = compute_multiclass_metrics(y_true, y_pred, num_classes=4)

    print(f"\nOverall Test Accuracy: {total_acc:.2f}%")
    print(f"Mean CPU Latency: {avg_latency_ms:.2f} ms/scan")
    print("\nConfusion Matrix (Rows: Ground Truth, Columns: Model Prediction):")
    print(f"{'Ground Truth':<22} | {'Glioma':<10} | {'Meningioma':<10} | {'Normal':<10} | {'Pituitary':<10}")
    print("-" * 72)
    for i, cls_name in enumerate(CLASS_DISPLAY):
        print(f"{cls_name:<22} | {cm[i][0]:<10d} | {cm[i][1]:<10d} | {cm[i][2]:<10d} | {cm[i][3]:<10d}")

    print("\nPer-Class Empirical Metrics:")
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall (Sens)':<14} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 72)
    for i, cls_name in enumerate(CLASS_DISPLAY):
        print(f"{cls_name:<22} | {prec[i]*100:6.2f}%    | {rec[i]*100:6.2f}%        | {f1[i]*100:6.2f}%    | {supp[i]:<8d}")

    macro_f1 = np.mean(f1) * 100.0
    macro_prec = np.mean(prec) * 100.0
    macro_rec = np.mean(rec) * 100.0
    print("-" * 72)
    print(f"{'Macro Average':<22} | {macro_prec:6.2f}%    | {macro_rec:6.2f}%        | {macro_f1:6.2f}%    | {len(y_true):<8d}")

    return {
        "architecture": "EfficientNet-B4",
        "test_samples": len(y_true),
        "overall_accuracy": round(float(total_acc), 2),
        "macro_precision": round(float(macro_prec), 2),
        "macro_recall": round(float(macro_rec), 2),
        "macro_f1": round(float(macro_f1), 2),
        "mean_latency_ms": round(float(avg_latency_ms), 2),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls_name: {
                "precision": round(float(prec[i] * 100), 2),
                "recall": round(float(rec[i] * 100), 2),
                "f1_score": round(float(f1[i] * 100), 2),
                "support": int(supp[i])
            }
            for i, cls_name in enumerate(CLASS_DISPLAY)
        }
    }

def evaluate_lightweight(test_dir="Dataset/Testing", ckpt_path="models_checkpoint/lightweight_best.pth"):
    print("\n" + "="*70)
    print(" 2. EMPIRICAL EVALUATION: LightweightTumorCNN (Binary Normal vs Tumor)")
    print("="*70)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at: {ckpt_path}")
        return None

    device = torch.device("cpu")
    model = LightweightTumorCNN(num_classes=2)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_transforms = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_dir, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    y_true_binary = []
    y_pred_binary = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.time()
            outputs = model(images)
            latencies.append((time.time() - t0) / len(images))
            preds = (torch.sigmoid(outputs).squeeze(-1) > 0.5).long()
            
            # Map: label 2 (notumor) -> 0, others (0,1,3) -> 1
            binary_labels = torch.where(labels == 2, torch.tensor(0), torch.tensor(1))
            
            y_true_binary.extend(binary_labels.cpu().numpy())
            y_pred_binary.extend(preds.cpu().numpy())

    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    acc = (y_true_binary == y_pred_binary).mean() * 100.0
    avg_latency = np.mean(latencies) * 1000.0

    # Binary confusion matrix
    # Normal = 0, Tumor = 1
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    sensitivity = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sensitivity) / (prec + sensitivity) if (prec + sensitivity) > 0 else 0.0

    print(f"Binary Test Accuracy: {acc:.2f}%")
    print(f"Tumor Sensitivity (Recall): {sensitivity:.2f}% (TP={tp}, FN={fn})")
    print(f"Healthy Specificity (TNR): {specificity:.2f}% (TN={tn}, FP={fp})")
    print(f"Precision: {prec:.2f}% | Binary F1-Score: {f1:.2f}%")
    print(f"Mean CPU Latency: {avg_latency:.3f} ms/scan (< 1 ms edge target)")

    return {
        "architecture": "LightweightTumorCNN",
        "task": "Binary Normal vs Tumor",
        "accuracy": round(float(acc), 2),
        "sensitivity": round(float(sensitivity), 2),
        "specificity": round(float(specificity), 2),
        "precision": round(float(prec), 2),
        "f1_score": round(float(f1), 2),
        "latency_ms": round(float(avg_latency), 3),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }

def main():
    eff_res = evaluate_efficientnet()
    lw_res = evaluate_lightweight()
    
    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cohort_size": 2800,
        "efficientnet_b4": eff_res,
        "lightweight_cnn": lw_res
    }

    with open("evaluation_testset_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("   EVALUATION COMPLETE: Results exported to evaluation_testset_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
