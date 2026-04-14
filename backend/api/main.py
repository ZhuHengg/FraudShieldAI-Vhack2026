from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from api.schemas import (
    DashboardStats, TransactionRequest, RiskResponse, EnsembleSHAPResponse,
    TopFeature, TransactionLogCreate, TransactionLogResponse,
    AnalystFeedback, FeedbackStatsResponse, RetrainRequest, RetrainResponse
)
from api.inference import EnsembleEngine
from api.database import get_db
from api.models import TransactionLog
import pandas as pd
import asyncio
import time
import logging
from datetime import datetime, timezone
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

    # Initialize DB tables (creates them if they don't exist)
    try:
        from api.database import init_db, engine as db_engine, SQLALCHEMY_DATABASE_URL
        init_db()
        logger.info(f"Database initialized ({SQLALCHEMY_DATABASE_URL[:30]}...)")

        # For existing tables, try to add new columns (safe to fail if they already exist)
        from sqlalchemy import text
        new_cols = {
            'analyst_label': 'VARCHAR' if 'sqlite' not in SQLALCHEMY_DATABASE_URL else 'TEXT',
            'analyst_notes': 'VARCHAR' if 'sqlite' not in SQLALCHEMY_DATABASE_URL else 'TEXT',
            'labeled_at': 'VARCHAR' if 'sqlite' not in SQLALCHEMY_DATABASE_URL else 'TEXT',
        }
        with db_engine.connect() as conn:
            for col_name, col_type in new_cols.items():
                try:
                    conn.execute(text(f'ALTER TABLE transaction_logs ADD COLUMN {col_name} {col_type}'))
                    logger.info(f"Added column '{col_name}' to transaction_logs")
                except Exception:
                    pass  # Column already exists
            conn.commit()
    except Exception as e:
        logger.warning(f"DB init: {e}")

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
    logger.info(f"Health check called. Engine is: {engine}, id: {id(engine)}")
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

@app.post("/api/v1/reset-stats")
async def reset_stats(request: Request):
    async with request.app.state.stats_lock:
        request.app.state.stats = {"total": 0, "approve": 0, "flag": 0, "block": 0, "latency_sum": 0.0}
    return {"status": "success", "message": "Stats reset"}

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

@app.post("/api/v1/transactions", status_code=201)
def save_transaction(transaction: TransactionLogCreate, db: Session = Depends(get_db)):
    """
    Save a scored transaction to the database.
    Silently ignores duplicates (by transaction_id).
    """
    try:
        db_txn = TransactionLog(**transaction.model_dump())
        db.add(db_txn)
        db.commit()
        return {"status": "success", "message": "Transaction saved"}
    except IntegrityError:
        # Transaction already exists, ignore
        db.rollback()
        return {"status": "success", "message": "Duplicate skipped"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save transaction: {str(e)}")

@app.get("/api/v1/transactions", response_model=list[TransactionLogResponse])
def get_transactions(limit: int = 200, db: Session = Depends(get_db)):
    """
    Fetch the most recent transactions from the database.
    """
    if limit > 500:
        limit = 500
    transactions = db.query(TransactionLog).order_by(TransactionLog.transaction_id.desc()).limit(limit).all()
    return transactions


# ══════════════════════════════════════════════════════════════════════════════
# CLOSED-LOOP RETRAINING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/feedback")
def save_feedback(feedback: AnalystFeedback, db: Session = Depends(get_db)):
    """
    Save analyst feedback (FRAUD/LEGIT label) for a transaction.
    This is the core of the closed-loop system.
    """
    if feedback.analyst_label not in ('FRAUD', 'LEGIT'):
        raise HTTPException(status_code=400, detail="analyst_label must be 'FRAUD' or 'LEGIT'")

    txn = db.query(TransactionLog).filter(
        TransactionLog.transaction_id == feedback.transaction_id
    ).first()

    if not txn:
        raise HTTPException(status_code=404, detail=f"Transaction '{feedback.transaction_id}' not found")

    txn.analyst_label = feedback.analyst_label
    txn.analyst_notes = feedback.analyst_notes
    txn.labeled_at = datetime.now(timezone.utc).isoformat()

    try:
        db.commit()
        return {
            "status": "success",
            "message": f"Transaction {feedback.transaction_id} labeled as {feedback.analyst_label}",
            "labeled_at": txn.labeled_at,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/feedback/stats", response_model=FeedbackStatsResponse)
def get_feedback_stats(db: Session = Depends(get_db)):
    """
    Get labeling progress statistics — how many transactions are
    labeled vs unlabeled, and whether we have enough to retrain.
    """
    from sqlalchemy import func

    total = db.query(func.count(TransactionLog.transaction_id)).scalar() or 0
    labeled = db.query(func.count(TransactionLog.transaction_id)).filter(
        TransactionLog.analyst_label.isnot(None)
    ).scalar() or 0
    fraud_labels = db.query(func.count(TransactionLog.transaction_id)).filter(
        TransactionLog.analyst_label == 'FRAUD'
    ).scalar() or 0
    legit_labels = db.query(func.count(TransactionLog.transaction_id)).filter(
        TransactionLog.analyst_label == 'LEGIT'
    ).scalar() or 0

    min_needed = 50
    return FeedbackStatsResponse(
        total_transactions=total,
        labeled_count=labeled,
        unlabeled_count=total - labeled,
        fraud_labels=fraud_labels,
        legit_labels=legit_labels,
        ready_to_retrain=labeled >= min_needed and fraud_labels >= 5 and legit_labels >= 5,
        min_samples_needed=min_needed,
    )


@app.post("/api/v1/retrain", response_model=RetrainResponse)
async def trigger_retrain(request: RetrainRequest, db: Session = Depends(get_db)):
    """
    Trigger closed-loop retraining using analyst-labeled transactions.
    Re-optimizes ensemble weights and thresholds, then hot-reloads the engine.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        from api.retrain import run_retrain
        result = run_retrain(engine, db, min_samples=request.min_labeled_samples)
        return RetrainResponse(**result)
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/api/v1/transactions/unlabeled", response_model=list[TransactionLogResponse])
def get_unlabeled_transactions(limit: int = 50, db: Session = Depends(get_db)):
    """
    Fetch transactions that have NOT been labeled by an analyst yet.
    Prioritizes high-risk transactions (ml_risk_score DESC) for efficient labeling.
    """
    if limit > 200:
        limit = 200
    txns = db.query(TransactionLog).filter(
        TransactionLog.analyst_label.is_(None)
    ).order_by(TransactionLog.ml_risk_score.desc()).limit(limit).all()
    return txns

@app.get("/api/v1/transactions/search", response_model=list[TransactionLogResponse])
def search_transactions(q: str = "", db: Session = Depends(get_db)):
    """
    Search transactions by ID, sender hash, or recipient hash.
    """
    if not q or len(q) < 3:
        return []
    
    search_term = f"%{q}%"
    transactions = db.query(TransactionLog).filter(
        (TransactionLog.transaction_id.ilike(search_term)) |
        (TransactionLog.user_hash.ilike(search_term)) |
        (TransactionLog.recipient_hash.ilike(search_term))
    ).order_by(TransactionLog.transaction_id.desc()).limit(50).all()
    return transactions
