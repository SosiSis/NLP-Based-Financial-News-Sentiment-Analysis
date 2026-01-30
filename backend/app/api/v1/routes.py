from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from ...services.predict_service import predict as predict_fn

router = APIRouter()


class PredictRequest(BaseModel):
    headline: str
    ticker: Optional[str] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None


class PredictResponse(BaseModel):
    prob_up: float
    label_up: bool
    finbert_positive: float
    finbert_negative: float
    finbert_neutral: float
    vader_compound: float
    message: Optional[str] = None


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict next-day direction from headline + optional features."""
    return predict_fn(req.dict())
