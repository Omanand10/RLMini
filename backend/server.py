"""FastAPI backend for Smart Traffic Control RL Dashboard.

Provides REST endpoints and Server-Sent Events for:
- Running simulations (fixed-time and RL)
- Training RL agents
- Fetching results and comparison data
- Live metric streaming
- CSV export
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

os.environ.setdefault(
    "SUMO_HOME", "/opt/anaconda3/lib/python3.12/site-packages/sumo"
)

from rl_engine.engine import (
    RESULTS_DIR,
    evaluate_rl,
    run_comparison,
    run_fixed_time,
    train_rl,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Traffic Control RL API",
    version="1.0.0",
    description="API for RL-based traffic signal control with SUMO simulation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
class SimulationState:
    """Thread-safe simulation state management."""

    def __init__(self):
        self.is_running = False
        self.current_task: str | None = None
        self.progress: float = 0.0
        self.live_metrics: list[dict] = []
        self.results: dict[str, Any] = {}
        self.error: str | None = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.is_running = False
            self.current_task = None
            self.progress = 0.0
            self.live_metrics = []
            self.results = {}
            self.error = None

    def update_progress(self, controller: str, step: int, total: int, metric: dict):
        with self._lock:
            self.progress = step / max(total, 1) * 100
            self.current_task = controller
            self.live_metrics.append(metric)
            # Keep last 2000 points
            if len(self.live_metrics) > 2000:
                self.live_metrics = self.live_metrics[-2000:]


state = SimulationState()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    total_timesteps: int = 20000
    num_seconds: int = 3600


class EvalRequest(BaseModel):
    num_seconds: int = 3600


class ComparisonRequest(BaseModel):
    total_timesteps: int = 20000
    num_seconds: int = 3600


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Get current simulation/training status."""
    with state._lock:
        return {
            "is_running": state.is_running,
            "current_task": state.current_task,
            "progress": round(state.progress, 1),
            "error": state.error,
            "has_results": bool(state.results),
        }


@app.post("/api/fixed-time")
async def start_fixed_time(req: EvalRequest):
    """Run fixed-time baseline simulation."""
    if state.is_running:
        raise HTTPException(409, "A simulation is already running")

    state.reset()
    state.is_running = True
    state.current_task = "fixed_time"

    def _run():
        try:
            results = run_fixed_time(
                num_seconds=req.num_seconds,
                progress_callback=state.update_progress,
            )
            with state._lock:
                state.results = results
                state.progress = 100.0
        except Exception as e:
            with state._lock:
                state.error = str(e)
        finally:
            with state._lock:
                state.is_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Fixed-time simulation started"}


@app.post("/api/train")
async def start_training(req: TrainRequest):
    """Start RL training."""
    if state.is_running:
        raise HTTPException(409, "A simulation is already running")

    state.reset()
    state.is_running = True
    state.current_task = "training"

    def _run():
        try:
            results = train_rl(
                total_timesteps=req.total_timesteps,
                num_seconds=req.num_seconds,
                progress_callback=state.update_progress,
            )
            with state._lock:
                state.results = results
                state.progress = 100.0
        except Exception as e:
            with state._lock:
                state.error = str(e)
        finally:
            with state._lock:
                state.is_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "RL training started"}


@app.post("/api/evaluate")
async def start_evaluation(req: EvalRequest):
    """Evaluate trained RL agent."""
    if state.is_running:
        raise HTTPException(409, "A simulation is already running")

    model_path = Path(__file__).parent.parent / "models" / "dqn_traffic.zip"
    if not model_path.exists():
        raise HTTPException(404, "No trained model found. Train first.")

    state.reset()
    state.is_running = True
    state.current_task = "evaluation"

    def _run():
        try:
            results = evaluate_rl(
                num_seconds=req.num_seconds,
                progress_callback=state.update_progress,
            )
            with state._lock:
                state.results = results
                state.progress = 100.0
        except Exception as e:
            with state._lock:
                state.error = str(e)
        finally:
            with state._lock:
                state.is_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "RL evaluation started"}


@app.post("/api/comparison")
async def start_comparison(req: ComparisonRequest):
    """Run full comparison: fixed-time → training → RL evaluation."""
    if state.is_running:
        raise HTTPException(409, "A simulation is already running")

    state.reset()
    state.is_running = True
    state.current_task = "comparison"

    def _run():
        try:
            results = run_comparison(
                total_timesteps=req.total_timesteps,
                num_seconds=req.num_seconds,
                progress_callback=state.update_progress,
            )
            with state._lock:
                state.results = results
                state.progress = 100.0
        except Exception as e:
            with state._lock:
                state.error = str(e)
        finally:
            with state._lock:
                state.is_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Full comparison started"}


@app.post("/api/reset")
async def reset_state():
    """Reset all state."""
    state.reset()
    return {"message": "State reset"}


@app.get("/api/results")
async def get_results():
    """Get the latest results."""
    with state._lock:
        if not state.results:
            raise HTTPException(404, "No results available yet")
        return state.results


@app.get("/api/metrics/live")
async def get_live_metrics():
    """Get live metrics buffer."""
    with state._lock:
        return {
            "metrics": state.live_metrics[-200:],
            "total_count": len(state.live_metrics),
        }


@app.get("/api/stream")
async def stream_metrics():
    """Server-Sent Events stream for live metrics."""

    async def event_generator():
        last_count = 0
        while True:
            with state._lock:
                current_count = len(state.live_metrics)
                is_running = state.is_running
                progress = state.progress
                error = state.error
                task = state.current_task

                if current_count > last_count:
                    new_metrics = state.live_metrics[last_count:current_count]
                    last_count = current_count
                else:
                    new_metrics = []

            data = {
                "is_running": is_running,
                "progress": round(progress, 1),
                "current_task": task,
                "error": error,
                "new_metrics": new_metrics[-10:],  # At most 10 new points
            }
            yield f"data: {json.dumps(data)}\n\n"

            if not is_running and last_count > 0:
                # Send final update
                yield f"data: {json.dumps({'done': True, 'progress': 100.0})}\n\n"
                break

            if not is_running and error:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/export/csv/{controller}")
async def export_csv(controller: str):
    """Download CSV results file."""
    filename_map = {
        "fixed_time": "fixed_time_results.csv",
        "rl_dqn": "rl_dqn_results.csv",
        "comparison": "comparison.csv",
    }

    if controller not in filename_map:
        raise HTTPException(400, f"Unknown controller: {controller}")

    csv_path = RESULTS_DIR / filename_map[controller]
    if not csv_path.exists():
        raise HTTPException(404, f"No CSV file found for {controller}")

    return FileResponse(
        path=str(csv_path),
        filename=filename_map[controller],
        media_type="text/csv",
    )


@app.get("/api/export/report")
async def export_report():
    """Download comparison JSON report."""
    report_path = RESULTS_DIR / "comparison.json"
    if not report_path.exists():
        raise HTTPException(404, "No comparison report found. Run comparison first.")

    return FileResponse(
        path=str(report_path),
        filename="comparison_report.json",
        media_type="application/json",
    )
