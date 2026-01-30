from pathlib import Path
from typing import Dict, Any
try:
    import joblib
except Exception:  # pragma: no cover - allow runtime without joblib installed
    joblib = None

# Lazy imports for heavy ML libs (allow API to start without them)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax
except Exception:  # pragma: no cover - allow runtime without torch/transformers
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    softmax = None

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

_model = None
_tokenizer = None
_vader = None
_sklearn_model = None

def _root_dir() -> Path:
    # project root (two parents up from this file -> backend/app/services -> backend/app -> backend)
    # adjust if you place `ml/serving` elsewhere
    return Path(__file__).resolve().parents[3]

MODEL_DIR = _root_dir() / "ml" / "serving"
SKMODEL_PATH = MODEL_DIR / "lstm_model.keras"
#"lstm_model.keras"

def load_models():
    """Load FinBERT, VADER and optional sklearn model."""
    global _tokenizer, _model, _vader, _sklearn_model
    # FinBERT (only if transformers/torch are available)
    model_name = "ProsusAI/finbert"
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        _tokenizer = None
        _model = None
    else:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model.to(device)
            _model.eval()
        except Exception:
            _tokenizer = None
            _model = None
    # VADER
    nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
    # Optional sklearn model
    if SKMODEL_PATH.exists() and joblib is not None:
        try:
            _sklearn_model = joblib.load(SKMODEL_PATH)
        except Exception:
            _sklearn_model = None


def _finbert_scores(text: str) -> Dict[str, float]:
    if not text or _tokenizer is None or _model is None:
        return {"finbert_positive": 0.0, "finbert_negative": 0.0, "finbert_neutral": 0.0}
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0].cpu().numpy()
    scores = {}
    for idx, p in enumerate(probs):
        label = _model.config.id2label[idx].lower()
        scores[f"finbert_{label}"] = float(p)
    return scores


def _vader_compound(text: str) -> float:
    if not text or _vader is None:
        return 0.0
    try:
        return float(_vader.polarity_scores(text)["compound"])
    except Exception:
        return 0.0


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("headline", "")
    fin = _finbert_scores(text)
    vader = _vader_compound(text)
    # If sklearn model present, build feature vector (attempt common names)
    if _sklearn_model is not None:
        feat_names = getattr(_sklearn_model, "feature_names_in_", None)
        if feat_names is not None:
            import numpy as np
            vals = []
            for f in feat_names:
                if f.startswith("finbert_"):
                    vals.append(fin.get(f, 0.0))
                elif f == "vader_compound":
                    vals.append(vader)
                else:
                    vals.append(float(payload.get(f, 0.0)))
            probs = _sklearn_model.predict_proba([vals])[0]
            prob_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
            label_up = prob_up >= 0.5
            return {
                "prob_up": prob_up,
                "label_up": bool(label_up),
                **fin,
                "vader_compound": vader,
                "message": "prediction from sklearn model",
            }
    # Fallback simple heuristic
    score = fin.get("finbert_positive", 0.0) - fin.get("finbert_negative", 0.0) + 0.5 * vader
    prob_up = 1 / (1 + 2.71828 ** (-5 * score))  # scaled sigmoid heuristic
    return {
        "prob_up": float(prob_up),
        "label_up": bool(prob_up >= 0.5),
        **fin,
        "vader_compound": vader,
        "message": "heuristic prediction (no sklearn model found)",
    }
