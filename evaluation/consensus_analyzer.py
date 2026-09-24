import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

class MultiModelConsensusAnalyzer:
    """
    Multi-Model Consensus & Inter-Model Discrepancy Analyzer.
    
    Orchestrates simultaneous tri-model inference across distinct architectural paradigms:
    1. EfficientNet-B4 (Compound Scaling CNN)
    2. Vision Transformer (ViT-B/16 Self-Attention)
    3. DenseNet-121 (Iterative Feature Reuse CNN)
    
    Evaluates:
    - Soft Voting Ensemble probability distribution
    - Architectural Concordance (Unanimous, Majority, Divergent)
    - Inter-Model Discrepancy Index (Standard Deviation of Class Probabilities)
    - Dissenting Model Identification & Clinical Diagnostic Advisory
    """
    CLASSES = ["Glioma", "Meningioma", "Normal Tissue (No Tumor)", "Pituitary Adenoma"]

    def __init__(self):
        pass

    def evaluate(
        self,
        tensor_240: torch.Tensor,
        tensor_224: torch.Tensor,
        effnet_model: torch.nn.Module,
        vit_model: torch.nn.Module,
        densenet_model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Runs tri-model inference and calculates ensemble concordance & discrepancy.
        """
        effnet_model.eval()
        vit_model.eval()
        densenet_model.eval()

        with torch.no_grad():
            logits_eff = effnet_model(tensor_240)
            probs_eff = F.softmax(logits_eff, dim=1).cpu().numpy()[0]

            logits_vit = vit_model(tensor_224)
            probs_vit = F.softmax(logits_vit, dim=1).cpu().numpy()[0]

            logits_dense = densenet_model(tensor_224)
            probs_dense = F.softmax(logits_dense, dim=1).cpu().numpy()[0]

        # Individual Top Predictions
        idx_eff = int(np.argmax(probs_eff))
        idx_vit = int(np.argmax(probs_vit))
        idx_dense = int(np.argmax(probs_dense))

        pred_eff = self.CLASSES[idx_eff]
        pred_vit = self.CLASSES[idx_vit]
        pred_dense = self.CLASSES[idx_dense]

        # Soft Voting Ensemble Average
        ensemble_probs = (probs_eff + probs_vit + probs_dense) / 3.0
        winner_idx = int(np.argmax(ensemble_probs))
        winner_class = self.CLASSES[winner_idx]
        winner_confidence = float(ensemble_probs[winner_idx])
        is_tumor = (winner_idx != 2) # Index 2 is 'Normal Tissue'

        # Inter-Model Discrepancy (Standard Deviation per class across models)
        stacked = np.stack([probs_eff, probs_vit, probs_dense], axis=0) # Shape: (3, 4)
        std_per_class = np.std(stacked, axis=0) # Shape: (4,)
        discrepancy_score = float(np.mean(std_per_class))
        max_discrepancy_idx = int(np.argmax(std_per_class))
        max_discrepancy_class = self.CLASSES[max_discrepancy_idx]

        # Architectural Concordance Analysis
        predictions = [idx_eff, idx_vit, idx_dense]
        unique_preds, counts = np.unique(predictions, return_counts=True)
        max_count = int(np.max(counts))

        dissenting_models = []
        if max_count == 3:
            concordance_level = "UNANIMOUS"
            agreement_pct = 100.0
            concordance_status = "Unanimous Consensus (100% Agreement)"
            concordance_badge = "UNANIMOUS"
        elif max_count == 2:
            concordance_level = "MAJORITY"
            agreement_pct = 66.7
            majority_class_idx = unique_preds[np.argmax(counts)]
            concordance_status = f"Majority Consensus (66.7% Agreement on {self.CLASSES[majority_class_idx]})"
            concordance_badge = "MAJORITY"

            if idx_eff != majority_class_idx:
                dissenting_models.append(f"EfficientNet-B4 voted '{pred_eff}' ({probs_eff[idx_eff]*100:.1f}%)")
            if idx_vit != majority_class_idx:
                dissenting_models.append(f"ViT-B/16 voted '{pred_vit}' ({probs_vit[idx_vit]*100:.1f}%)")
            if idx_dense != majority_class_idx:
                dissenting_models.append(f"DenseNet-121 voted '{pred_dense}' ({probs_dense[idx_dense]*100:.1f}%)")
        else:
            concordance_level = "DIVERGENT"
            agreement_pct = 33.3
            concordance_status = "High Discrepancy (Divergent Model Verdicts)"
            concordance_badge = "DIVERGENT"
            dissenting_models = [
                f"EfficientNet-B4: '{pred_eff}'",
                f"ViT-B/16: '{pred_vit}'",
                f"DenseNet-121: '{pred_dense}'"
            ]

        # Clinical Reliability Rating
        if concordance_level == "UNANIMOUS" and discrepancy_score < 0.10:
            reliability = "TIER 1 (High Reliability - Triple Architecture Agreement)"
            reliability_color = "#34d399"
        elif concordance_level == "MAJORITY" and discrepancy_score < 0.20:
            reliability = "TIER 2 (Moderate Reliability - Dual Architecture Agreement)"
            reliability_color = "#38bdf8"
        else:
            reliability = "TIER 3 (Borderline Reliability - Discrepancy Alert)"
            reliability_color = "#f87171"

        # Clinical Action Protocol
        protocol_map = {
            0: "Intraparenchymal infiltration characteristic of Glioma. Neurosurgical oncology consultation advised.",
            1: "Extra-axial dural attachment characteristic of Meningioma. Neuro-oncology review and surgical assessment advised.",
            2: "No abnormal intracranial mass effect identified. Routine surveillance indicated.",
            3: "Sellar / suprasellar mass characteristic of Pituitary Adenoma. Comprehensive endocrinology panel required."
        }
        base_protocol = protocol_map.get(winner_idx, "Medical specialist review advised.")
        if concordance_level == "DIVERGENT":
            clinical_protocol = f"URGENT RADIOLOGY REVIEW: Model architectures produced divergent findings. {base_protocol}"
        elif len(dissenting_models) > 0:
            clinical_protocol = f"{base_protocol} Note: Discrepancy observed ({', '.join(dissenting_models)})."
        else:
            clinical_protocol = f"{base_protocol} Validated by triple-architecture unanimous agreement."

        # Structured per-class breakdown for UI display
        breakdown_rows = []
        for i, cname in enumerate(self.CLASSES):
            breakdown_rows.append({
                "class_name": cname,
                "effnet_prob": float(probs_eff[i]),
                "vit_prob": float(probs_vit[i]),
                "densenet_prob": float(probs_dense[i]),
                "ensemble_prob": float(ensemble_probs[i]),
                "std_dev": float(std_per_class[i])
            })

        return {
            "winner_class": winner_class,
            "winner_idx": winner_idx,
            "is_tumor": is_tumor,
            "confidence": winner_confidence,
            "prediction_label": f"POSITIVE: {winner_class.upper()}" if is_tumor else "NEGATIVE FOR INTRACRANIAL LESION (NORMAL)",
            "concordance_level": concordance_level,
            "concordance_status": concordance_status,
            "concordance_badge": concordance_badge,
            "agreement_pct": agreement_pct,
            "discrepancy_score": discrepancy_score,
            "max_discrepancy_class": max_discrepancy_class,
            "dissenting_models": dissenting_models,
            "reliability": reliability,
            "reliability_color": reliability_color,
            "clinical_protocol": clinical_protocol,
            "individual_votes": {
                "EfficientNet-B4": {
                    "class": pred_eff,
                    "confidence": float(probs_eff[idx_eff]),
                    "probabilities": probs_eff.tolist()
                },
                "Vision Transformer (ViT-B/16)": {
                    "class": pred_vit,
                    "confidence": float(probs_vit[idx_vit]),
                    "probabilities": probs_vit.tolist()
                },
                "DenseNet-121": {
                    "class": pred_dense,
                    "confidence": float(probs_dense[idx_dense]),
                    "probabilities": probs_dense.tolist()
                }
            },
            "ensemble_probabilities": ensemble_probs.tolist(),
            "breakdown_rows": breakdown_rows
        }
