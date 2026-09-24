import io
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    class_id: int
    confidence_score: float
    recommended_clinical_action: str
    probabilities: Dict[str, float]

class TumorPredictor:
    """
    Modul inferensi independen (deep module).
    Menyembunyikan detail: preprocessing tensor, normalisasi channel,
    evaluasi forward model, dan pemetaan rekomendasi klinis.
    """
    CLASS_NAMES = ("Normal", "Glioma", "Meningioma", "Tumor Hipofisis")

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _prepare_tensor(self, image_bytes: bytes) -> torch.Tensor:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        resized = cv2.resize(img_np, (380, 380))

        # 4-channel input array (T1, T1ce, T2, FLAIR)
        tensor_data = np.zeros((4, 380, 380), dtype=np.float32)
        tensor_data[0] = resized[:, :, 0] / 255.0
        tensor_data[1] = resized[:, :, 1] / 255.0
        tensor_data[2] = resized[:, :, 2] / 255.0
        tensor_data[3] = (tensor_data[0] + tensor_data[1]) / 2.0

        return torch.from_numpy(tensor_data).unsqueeze(0).to(self.device)

    def _get_clinical_action(self, class_id: int, confidence: float) -> str:
        if confidence < 0.50:
            return "Hasil Tidak Pasti (Borderline): Diperlukan peninjauan manual oleh Dokter Spesialis Radiologi."

        actions = {
            0: "Normal: Tidak terdeteksi massa, evaluasi rutin berkala.",
            1: "Glioma Terdeteksi: Rujukan segera ke Dokter Spesialis Bedah Saraf (Onkologi).",
            2: "Meningioma Terdeteksi: Konsultasi Neurologi dan pemantauan perkembangan massa disarankan.",
            3: "Tumor Hipofisis Terdeteksi: Rujukan evaluasi fungsi hormon ke Tim Endokrinologi dan Bedah Saraf."
        }
        return actions.get(class_id, "Tinjauan medis lanjutan.")

    def predict(self, image_bytes: bytes) -> PredictionResult:
        tensor_input = self._prepare_tensor(image_bytes)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        label = self.CLASS_NAMES[pred_id]
        action = self._get_clinical_action(pred_id, confidence)

        prob_dict = {name: float(probs[i]) for i, name in enumerate(self.CLASS_NAMES)}

        return PredictionResult(
            predicted_class=label,
            class_id=pred_id,
            confidence_score=confidence,
            recommended_clinical_action=action,
            probabilities=prob_dict
        )
