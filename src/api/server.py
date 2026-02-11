"""FastAPI server for arbitrage and ML prediction endpoints.

Run with: uvicorn src.api.server:app --reload --port 8000
"""

import sys
sys.path.insert(0, '.')

import asyncio
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import polycast modules
from src.polycast.cross_arb import find_cross_market_arbitrage
from src.polycast.scanner import scan_polymarket_raw
from src.ml.inference_engine import InferenceEngine
from src.ml.feature_builder_v2 import AdvancedFeatureBuilder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_PATH = DATA_DIR / "models" / "ensemble_model.pkl"


# ============================================================
# Pydantic Models
# ============================================================

class ArbitrageOpportunity(BaseModel):
    id: str
    type: str  # "cross_platform" or "intra_market"
    platform_a: str
    platform_b: Optional[str] = None
    question: str
    price_a: float
    price_b: Optional[float] = None
    edge_pct: float
    ml_score: Optional[float] = None
    confidence: Optional[str] = None
    timestamp: float


class MLPrediction(BaseModel):
    market_id: str
    question: str
    platform: str
    yes_price: float
    no_price: float
    ml_score: float
    confidence: str
    edge_pct: float
    recommendation: str
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    model_auc: Optional[float] = None


# ============================================================
# Global State
# ============================================================

class AppState:
    def __init__(self):
        self.model_data: Optional[Dict] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.websocket_clients: Set[WebSocket] = set()
        self.last_arbitrage_data: List[Dict] = []
        self.last_update_time: float = 0

    def load_model(self):
        """Load ML model from disk."""
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model_data = pickle.load(f)
            self.inference_engine = InferenceEngine()
            return True
        return False


state = AppState()


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading ML model...")
    if state.load_model():
        print(f"Model loaded: v{state.model_data.get('version', 'unknown')}")
        print(f"Model AUC: {state.model_data.get('metrics', {}).get('auc', 'N/A')}")
    else:
        print("WARNING: No model found at", MODEL_PATH)

    # Start background task for WebSocket broadcasts
    broadcast_task = asyncio.create_task(broadcast_updates())

    yield

    # Shutdown
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Polycast Arbitrage API",
    description="ML-powered arbitrage detection for prediction markets",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REST Endpoints
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model_data is not None,
        model_version=state.model_data.get("version") if state.model_data else None,
        model_auc=state.model_data.get("metrics", {}).get("auc") if state.model_data else None,
    )


@app.get("/api/arbitrage", response_model=List[ArbitrageOpportunity])
async def get_arbitrage_opportunities(
    min_edge: float = Query(default=0.01, description="Minimum edge percentage (0.01 = 1%)"),
    limit: int = Query(default=50, description="Maximum results to return"),
    include_ml_scores: bool = Query(default=True, description="Include ML predictions"),
):
    """
    Get current arbitrage opportunities.

    Returns cross-platform and intra-market arbitrage opportunities
    ranked by edge percentage and ML confidence.
    """
    opportunities = []

    try:
        # Get cross-platform arbitrage
        cross_arb = await asyncio.to_thread(find_cross_market_arbitrage)

        for arb in cross_arb:
            edge = arb.get("profit_pct", 0) / 100  # Convert to decimal
            if edge < min_edge:
                continue

            opp = ArbitrageOpportunity(
                id=f"cross_{arb.get('poly_id', '')}_{arb.get('kalshi_id', '')}",
                type="cross_platform",
                platform_a="polymarket",
                platform_b="kalshi",
                question=arb.get("poly_question", "")[:200],
                price_a=arb.get("poly_yes", 0.5),
                price_b=arb.get("kalshi_yes", 0.5),
                edge_pct=edge,
                ml_score=None,
                confidence=None,
                timestamp=time.time(),
            )

            # Add ML score if requested and model available
            if include_ml_scores and state.inference_engine:
                try:
                    score = await get_ml_score_for_market({
                        "id": arb.get("poly_id", ""),
                        "question": arb.get("poly_question", ""),
                        "yes_price": arb.get("poly_yes", 0.5),
                        "no_price": 1 - arb.get("poly_yes", 0.5),
                        "volume": 0,
                    })
                    opp.ml_score = score.get("score")
                    opp.confidence = score.get("confidence")
                except Exception:
                    pass

            opportunities.append(opp)

        # Get intra-market opportunities (YES + NO > 1)
        raw_markets = await asyncio.to_thread(scan_polymarket_raw)

        for market in raw_markets:
            yes_price = market.get("yes_price", 0.5)
            no_price = market.get("no_price", 0.5)
            overround = yes_price + no_price - 1

            if overround > min_edge:
                opp = ArbitrageOpportunity(
                    id=f"intra_{market.get('id', '')}",
                    type="intra_market",
                    platform_a="polymarket",
                    platform_b=None,
                    question=market.get("question", "")[:200],
                    price_a=yes_price,
                    price_b=no_price,
                    edge_pct=overround,
                    ml_score=None,
                    confidence=None,
                    timestamp=time.time(),
                )

                if include_ml_scores and state.inference_engine:
                    try:
                        score = await get_ml_score_for_market(market)
                        opp.ml_score = score.get("score")
                        opp.confidence = score.get("confidence")
                    except Exception:
                        pass

                opportunities.append(opp)

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge_pct, reverse=True)

        # Cache for WebSocket broadcasts
        state.last_arbitrage_data = [o.model_dump() for o in opportunities[:limit]]
        state.last_update_time = time.time()

        return opportunities[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions", response_model=List[MLPrediction])
async def get_ml_predictions(
    min_score: float = Query(default=0.5, description="Minimum ML score (0-1)"),
    limit: int = Query(default=50, description="Maximum results"),
):
    """
    Get ML-scored market predictions.

    Returns markets ranked by ML opportunity score.
    """
    if not state.inference_engine:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        raw_markets = await asyncio.to_thread(scan_polymarket_raw)
        predictions = []

        for market in raw_markets:
            try:
                score_result = await get_ml_score_for_market(market)

                if score_result["score"] >= min_score:
                    predictions.append(MLPrediction(
                        market_id=market.get("id", ""),
                        question=market.get("question", "")[:200],
                        platform="polymarket",
                        yes_price=market.get("yes_price", 0.5),
                        no_price=market.get("no_price", 0.5),
                        ml_score=score_result["score"],
                        confidence=score_result["confidence"],
                        edge_pct=score_result.get("edge", 0),
                        recommendation=score_result.get("recommendation", "hold"),
                        timestamp=time.time(),
                    ))
            except Exception:
                continue

        predictions.sort(key=lambda x: x.ml_score, reverse=True)
        return predictions[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get current ML model performance metrics."""
    if not state.model_data:
        raise HTTPException(status_code=503, detail="No model loaded")

    return {
        "version": state.model_data.get("version"),
        "timestamp": state.model_data.get("timestamp"),
        "training_samples": state.model_data.get("training_samples"),
        "snapshot_count": state.model_data.get("snapshot_count"),
        "metrics": state.model_data.get("metrics", {}),
    }


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws/arbitrage")
async def websocket_arbitrage(websocket: WebSocket):
    """
    WebSocket endpoint for real-time arbitrage updates.

    Sends updates every 30 seconds with latest opportunities.
    Client can also send {"action": "refresh"} to force immediate update.
    """
    await websocket.accept()
    state.websocket_clients.add(websocket)

    try:
        # Send initial data
        if state.last_arbitrage_data:
            await websocket.send_json({
                "type": "arbitrage_update",
                "data": state.last_arbitrage_data,
                "timestamp": state.last_update_time,
            })

        # Listen for client messages
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)

                if message.get("action") == "refresh":
                    # Force refresh
                    await get_arbitrage_opportunities()
                    await websocket.send_json({
                        "type": "arbitrage_update",
                        "data": state.last_arbitrage_data,
                        "timestamp": state.last_update_time,
                    })
            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect:
        pass
    finally:
        state.websocket_clients.discard(websocket)


async def broadcast_updates():
    """Background task to broadcast updates to all WebSocket clients."""
    while True:
        await asyncio.sleep(30)  # Update every 30 seconds

        if state.websocket_clients:
            try:
                # Refresh data
                await get_arbitrage_opportunities()

                # Broadcast to all clients
                message = {
                    "type": "arbitrage_update",
                    "data": state.last_arbitrage_data,
                    "timestamp": state.last_update_time,
                }

                disconnected = set()
                for client in state.websocket_clients:
                    try:
                        await client.send_json(message)
                    except Exception:
                        disconnected.add(client)

                state.websocket_clients -= disconnected

            except Exception as e:
                print(f"Broadcast error: {e}")


# ============================================================
# Helper Functions
# ============================================================

async def get_ml_score_for_market(market: Dict) -> Dict:
    """Get ML score for a single market."""
    if not state.inference_engine:
        return {"score": 0.5, "confidence": "unknown", "edge": 0, "recommendation": "hold"}

    try:
        # Build features
        builder = AdvancedFeatureBuilder()
        features = builder.build_features({
            "id": market.get("id", ""),
            "question": market.get("question", ""),
            "description": market.get("description", ""),
            "yes_price": market.get("yes_price", 0.5),
            "no_price": market.get("no_price", 0.5),
            "volume": market.get("volume", 0),
            "category": market.get("category", ""),
            "close_time": market.get("close_time", time.time() + 86400 * 7),
            "created_at": market.get("created_at", time.time() - 86400 * 30),
        })

        # Get prediction from ensemble
        ensemble = state.model_data["ensemble"]
        X = features.to_vector().reshape(1, -1)
        score = float(ensemble.predict_proba(X)[0])

        # Determine confidence
        if score >= 0.75:
            confidence = "high"
            recommendation = "buy"
        elif score >= 0.55:
            confidence = "medium"
            recommendation = "consider"
        else:
            confidence = "low"
            recommendation = "hold"

        # Calculate edge
        yes_price = market.get("yes_price", 0.5)
        edge = score - yes_price if score > 0.5 else 0

        return {
            "score": score,
            "confidence": confidence,
            "edge": edge,
            "recommendation": recommendation,
        }

    except Exception as e:
        return {"score": 0.5, "confidence": "error", "edge": 0, "recommendation": "hold"}


# ============================================================
# Run with: uvicorn src.api.server:app --reload --port 8000
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
