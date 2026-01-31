from fastapi import FastAPI
from .api.v1.routes import router
from .services import predict_service

app = FastAPI(title="Financial News Predictor")
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup():
    try:
        predict_service.load_models()
    except Exception:
        
        pass
