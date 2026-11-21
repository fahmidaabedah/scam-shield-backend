import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .scam_detect import Detector

MODEL_PATH = Path(__file__).parent / 'models' / 'tfidf-lr.joblib'
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]

app = FastAPI(title='Scam Message Detector API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['POST', 'GET'],
    allow_headers=['*'],
)

detector = Detector(MODEL_PATH)


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ClassifyResponse(BaseModel):
    verdict: str
    scam_score: float
    type: str
    type_bn: str
    probability: float
    all: dict[str, float]
    why: list[str]


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/classify', response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail='message is empty')
    result = detector.classify(msg)
    why = [w for w, _ in detector.explain(msg)] if result['verdict'] == 'SCAM' else []
    return {**result, 'why': why}
