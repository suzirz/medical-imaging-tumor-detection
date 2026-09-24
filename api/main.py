import os
import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.efficientnet_tumor_classifier import BrainTumorClassifier
from evaluation.gradcam_visualizer import GradCAMVisualizer

app = FastAPI(
    title="Medical Imaging Tumor Detection API",
    description="FastAPI service untuk inferensi model CNN EfficientNet-B4 + Grad-CAM Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Model & Helper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorClassifier(num_classes=4, pretrained=False)
model.to(device)
model.eval()

visualizer = GradCAMVisualizer(model)
CLASS_NAMES = ["Normal", "Glioma", "Meningioma", "Tumor Hipofisis"]

class PredictionResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence_score: float
    recommended_clinical_action: str
    probabilities: dict

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(device), "model": "EfficientNet-B4-4Ch"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_tumor(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(pil_img)
        
        # Resize ke (380, 380)
        resized = cv2.resize(img_np, (380, 380))
        # Mengubah jadi tensor 4 channel (simulasi 4 modalitas dari input 3-channel + extra channel)
        ch4 = np.zeros((4, 380, 380), dtype=np.float32)
        ch4[0] = resized[:, :, 0] / 255.0
        ch4[1] = resized[:, :, 1] / 255.0
        ch4[2] = resized[:, :, 2] / 255.0
        ch4[3] = (ch4[0] + ch4[1]) / 2.0  # FLAIR approximation
        
        tensor_input = torch.from_numpy(ch4).unsqueeze(0).to(device) # (1, 4, 380, 380)

        with torch.no_grad():
            logits = model(tensor_input)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        pred_label = CLASS_NAMES[pred_id]
        action = visualizer.get_clinical_action(pred_id, confidence)

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return PredictionResponse(
            predicted_class=pred_label,
            class_id=pred_id,
            confidence_score=confidence,
            recommended_clinical_action=action,
            probabilities=prob_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
