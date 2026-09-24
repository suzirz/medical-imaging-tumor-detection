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
    icd_code: str = "ICD-10: C71.9 / ICD-O-3: 9380/3"
    snomed_code: str = "SNOMED-CT: 126952004"

class TumorPredictor:
    """
    Modul inferensi independen (deep module).
    Menyembunyikan detail: preprocessing tensor, normalisasi channel,
    evaluasi forward model, dan pemetaan rekomendasi klinis serta ontologi ICD/SNOMED.
    """
    CLASS_NAMES = ("Normal", "Glioma", "Meningioma", "Tumor Hipofisis")
    ICD_CODES = {
        0: "ICD-10: Z00.00 (Pemeriksaan Medis Rutin Tanpa Kelainan)",
        1: "ICD-10: C71.9 / ICD-O-3: 9380/3 (Neoplasma Ganas Otak / Glioma)",
        2: "ICD-10: D32.9 / ICD-O-3: 9530/0 (Neoplasma Jinak Selaput Otak / Meningioma)",
        3: "ICD-10: D35.2 / ICD-O-3: 8272/0 (Neoplasma Jinak Kelenjar Hipofisis)"
    }
    SNOMED_CODES = {
        0: "SNOMED-CT: 17621005 (Normal Clinical Finding)",
        1: "SNOMED-CT: 126952004 (Neoplasm of Brain / Malignant Glioma)",
        2: "SNOMED-CT: 1947003 (Meningioma of Brain)",
        3: "SNOMED-CT: 254956000 (Pituitary Adenoma)"
    }

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

        icd = self.ICD_CODES.get(pred_id, "ICD-10: R90.82")
        snomed = self.SNOMED_CODES.get(pred_id, "SNOMED-CT: 404684003")

        return PredictionResult(
            predicted_class=label,
            class_id=pred_id,
            confidence_score=confidence,
            recommended_clinical_action=action,
            probabilities=prob_dict,
            icd_code=icd,
            snomed_code=snomed
        )
