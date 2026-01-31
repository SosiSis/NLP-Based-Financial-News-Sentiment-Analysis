from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from tensorflow.keras.models import load_model


_finbert_model = None
_tokenizer = None
_vader = None
_lstm_model = None



def _root_dir() -> Path:
   
    return Path(__file__).resolve().parents[3]


MODEL_DIR = _root_dir() / "ml" / "serving"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"



def load_models():
    global _finbert_model, _tokenizer, _vader, _lstm_model

    print("Model directory:", MODEL_DIR)
    print("LSTM model path:", LSTM_MODEL_PATH)

    # -------- FinBERT --------
    model_name = "ProsusAI/finbert"
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _finbert_model.to(device)
        _finbert_model.eval()
        print("FinBERT loaded on", device)
    except Exception as e:
        print("Failed to load FinBERT:", e)
        _finbert_model = None
        _tokenizer = None

    # -------- VADER --------
    nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
    print("VADER loaded")

    # -------- Keras LSTM --------
    if LSTM_MODEL_PATH.exists():
        try:
            _lstm_model = load_model(LSTM_MODEL_PATH)
            print("Keras LSTM model loaded")
        except Exception as e:
            print("Failed to load LSTM model:", e)
            _lstm_model = None
    else:
        print("LSTM model NOT FOUND")



def _finbert_scores(text: str) -> Dict[str, float]:
    if not text or _tokenizer is None or _finbert_model is None:
        return {
            "finbert_positive": 0.0,
            "finbert_negative": 0.0,
            "finbert_neutral": 0.0,
        }

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    device = next(_finbert_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()

    scores = {}
    for i, p in enumerate(probs):
        label = _finbert_model.config.id2label[i].lower()
        scores[f"finbert_{label}"] = float(p)

    return scores



def _vader_compound(text: str) -> float:
    if not text or _vader is None:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])



def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("headline", "")

    fin = _finbert_scores(text)
    vader = _vader_compound(text)

  
    if _lstm_model is not None:
        # Feature order MUST match training
        features = np.array([[
    fin.get("finbert_positive", 0.0),
    fin.get("finbert_negative", 0.0),
    fin.get("finbert_neutral", 0.0),
    vader,
    float(payload.get("Open", 0.0)),
    float(payload.get("High", 0.0)),
    float(payload.get("Low", 0.0)),
    float(payload.get("Close", 0.0)),
    float(payload.get("Volume", 0.0)),
    float(payload.get("SMA_5", 0.0)),
]])


        # LSTM expects 3D input
        features = features.reshape((1, 1, features.shape[1]))

        prob_up = float(_lstm_model.predict(features, verbose=0)[0][0])

        return {
            "prob_up": prob_up,
            "label_up": prob_up >= 0.5,
            **fin,
            "vader_compound": vader,
            "message": "prediction from Keras LSTM model",
        }

   
    score = (
        fin.get("finbert_positive", 0.0)
        - fin.get("finbert_negative", 0.0)
        + 0.5 * vader
    )
    prob_up = 1 / (1 + np.exp(-5 * score))

    return {
        "prob_up": float(prob_up),
        "label_up": prob_up >= 0.5,
        **fin,
        "vader_compound": vader,
        "message": "heuristic prediction (LSTM model not found)",
    }
