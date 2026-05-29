#!/usr/bin/env python3
"""
predictor_service.py — IRT Parameter Predictor HTTP Sidecar

Wraps IRTPredictor as a FastAPI service so the TypeScript oracle service
can call it without reloading the ML model on every question.

Usage:
    cd /Users/gabib/git/thesis/PoCW-IRT-Calibrator
    OMP_NUM_THREADS=1 python3 predictor_service.py

    # Or with explicit port:
    OMP_NUM_THREADS=1 python3 predictor_service.py --port 3001

The oracle service calls POST http://localhost:3001/predict
"""

import argparse
import os
import sys
from pathlib import Path

IRT_RUN_DIR = Path(__file__).parent / "irt_runs" / "20260525_2339_12000q_12m_4pl_mmlu-boolq-triviaqa"

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[predictor_service] Missing dependencies. Run: pip install fastapi uvicorn pydantic")
    sys.exit(1)

sys.path.insert(0, str(IRT_RUN_DIR))
from irt_predictor import IRTPredictor  # type: ignore

app = FastAPI(title="IRT Predictor Sidecar")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"])

predictor: IRTPredictor | None = None


class PredictRequest(BaseModel):
    question: str
    choices: list[str] = []
    theta: float = 0.0


class PredictResponse(BaseModel):
    a: float
    b: float
    c: float
    d: float
    p_correct: float
    difficulty: str
    discrimination: str


@app.on_event("startup")
def load_model():
    global predictor
    print("[predictor_service] Loading IRT predictor model…")
    predictor = IRTPredictor(model_dir=str(IRT_RUN_DIR))
    print("[predictor_service] Model loaded.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(req.question, req.choices, req.theta)
    return PredictResponse(**result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
