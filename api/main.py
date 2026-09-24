import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

from models.efficientnet_tumor_classifier import BrainTumorClassifier
from models.predictor import TumorPredictor, PredictionResult

app = FastAPI(
    title="Medical Imaging Tumor Detection API",
    description="Layanan inferensi model klasifikasi tumor otak berbasis EfficientNet-B4.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_predictor() -> TumorPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorClassifier(num_classes=4, pretrained=False)
    return TumorPredictor(model=model, device=device)

class PredictionResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence_score: float
    recommended_clinical_action: str
    probabilities: Dict[str, float]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_tumor(
    file: UploadFile = File(...),
    predictor: TumorPredictor = Depends(get_predictor)
):
    try:
        content = await file.read()
        res: PredictionResult = predictor.predict(content)
        return PredictionResponse(
            predicted_class=res.predicted_class,
            class_id=res.class_id,
            confidence_score=res.confidence_score,
            recommended_clinical_action=res.recommended_clinical_action,
            probabilities=res.probabilities
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {str(err)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
