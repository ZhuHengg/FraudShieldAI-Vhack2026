from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import DashboardStats, TransactionRequest, RiskResponse, EnsembleSHAPResponse, TopFeature
from api.inference import EnsembleEngine
import pandas as pd
import asyncio
import time
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global engine
    logger.info("Initializing Application and loading models...")
    app.state.stats_lock = asyncio.Lock()
    app.state.stats = {"total": 0, "approve": 0, "flag": 0, "block": 0, "latency_sum": 0.0}
    
    try:
        # Load the unified inference engine
        engine = EnsembleEngine(
            iso_model_dir='models/unsupervised/isolation_forest/outputs/model',
            lgb_model_dir='models/supervised/outputs/model',
            ensemble_dir='models/ensemble/outputs/model'
        )
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
    yield
    # shutdown (add cleanup here if needed)

app = FastAPI(
    title="FraudShield AI API",
    description="Real-Time Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}

@app.get("/api/v1/config")
def get_config():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {
        "approve_threshold": engine.approve_threshold,
        "flag_threshold":    engine.flag_threshold,
        "weights": {
            "lgb": engine.w_lgb,
            "iso": engine.w_iso,
            "beh": engine.w_beh,
        }
    }

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
@app.post("/api/v1/score-transaction", response_model=RiskResponse)
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

@app.post("/api/v1/explain/{transaction_id}", response_model=EnsembleSHAPResponse)
async def explain_transaction(transaction_id: str, transaction: TransactionRequest):
    """
    Generate SHAP explanations for a single transaction using the already loaded EnsembleEngine.
    Combines LightGBM SHAP values, Isolation Forest anomaly components, and Behavioral weights.
    """
    if not engine or not engine.explainer:
        raise HTTPException(status_code=503, detail="Risk Engine or SHAP explainer not fully initialized")

    try:
        # 1. Prepare data using shared preprocessing
        df = engine.preprocess(transaction.model_dump())
        X_lgb = engine.get_lgb_features(df)
        X_iso = engine.get_iso_features(df)
        
        # 2. Get Layer Scores
        # LightGBM score (probability * 100)
        lgb_score = engine.score_lgb(X_lgb)
        # Isolation Forest score (scaled [0, 100])
        iso_score = engine.score_iso(X_iso)
        # Behavioral score (scaled [0, 100])
        beh_score, _ = engine.score_beh(df)
        
        # 3. Get LightGBM SHAP values
        # shap_values is an Explanation object
        shap_explanation = engine.explainer(X_lgb)
        lgb_base_value = float(shap_explanation.base_values[0])
        lgb_contributions = shap_explanation.values[0]
        
        # 4. Get Ensemble Weights
        w_lgb = engine.w_lgb
        w_iso = engine.w_iso
        w_beh = engine.w_beh
            
        # 5. Compute Weighted Composite SHAP
        # LightGBM contributes its SHAP values * its ensemble weight
        weighted_contributions = {
            feat: float(val) * w_lgb 
            for feat, val in zip(X_lgb.columns, lgb_contributions)
        }
        
        # Add proxy contributions for Unsupervised and Behavioral layers
        # (Since they don't have native SHAP, we represent their overall impact as a single feature bar)
        weighted_contributions.update({
            "isolation_forest_anomaly": (iso_score / 100.0) * w_iso,
            "behavioral_rules": (beh_score / 100.0) * w_beh
        })
        
        # 6. Extract Top 5 Features
        # Using list() explicitly to satisfy potential type checker ambiguities
        all_features = [TopFeature(feature=k, contribution=v) for k, v in weighted_contributions.items()]
        top_features = sorted(
            all_features,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
        # Return at most 5
        top_features = top_features[:5]
        
        return EnsembleSHAPResponse(
            transaction_id=transaction_id,
            base_value=float(lgb_base_value * w_lgb),
            iso_score=float(iso_score / 100.0),
            lgb_score=float(lgb_score / 100.0),
            beh_score=float(beh_score / 100.0),
            top_features=top_features
        )
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
