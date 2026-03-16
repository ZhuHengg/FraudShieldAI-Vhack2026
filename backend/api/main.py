from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import DashboardStats, TransactionRequest, RiskResponse
from api.inference import EnsembleEngine
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FraudShield AI API",
    description="Real-Time Fraud Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Initializing Application and loading models...")
    app.state.stats_lock = asyncio.Lock()
    app.state.stats = {
        "total": 0,
        "approve": 0,
        "flag": 0,
        "block": 0,
        "latency_sum": 0.0
    }
    
    try:
        # Load the unified inference engine
        engine = EnsembleEngine(
            iso_model_dir='models/unsupervised/isolation_forest/outputs/model',
            lgb_model_dir='models/supervised/outputs/model',
            ensemble_dir='models/ensemble/outputs/model'
        )
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}

@app.get("/api/v1/stats", response_model=DashboardStats)
async def get_dashboard_stats(request: Request):
    async with request.app.state.stats_lock:
        stats = request.app.state.stats
        total = max(stats["total"], 1)
        fraud_rate = (stats["block"] + stats["flag"]) / total * 100
        
        return DashboardStats(
            total_transactions=stats["total"],
            approved=stats["approve"],
            flagged=stats["flag"],
            blocked=stats["block"],
            avg_latency_ms=stats["latency_sum"] / total,
            fraud_rate_estimate=round(fraud_rate, 2)
        )

@app.post("/predict", response_model=RiskResponse)
async def predict_fraud(transaction: TransactionRequest, request: Request):
    """
    Predict fraud risk for a given transaction.
    Returns a detailed risk response.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Risk Engine not fully initialized")
        
    start_time = time.time()
    
    result = engine.predict(transaction)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Map risk_level to the stats format (approve/flag/block)
    decision_map = {
        "low": "approve",
        "medium": "flag",
        "high": "block"
    }
    
    decision = decision_map.get(result.risk_level.lower(), "flag")
    
    # Update global thread-safe state
    async with request.app.state.stats_lock:
        request.app.state.stats["total"] += 1
        request.app.state.stats[decision] += 1
        request.app.state.stats["latency_sum"] += latency_ms

    return result
