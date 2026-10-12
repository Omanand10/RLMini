"""Training and evaluation engine for RL traffic signal control.

Supports:
- Training with Stable-Baselines3 DQN
- Fixed-time baseline evaluation
- Side-by-side comparison with CSV export
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "SUMO_HOME", "/opt/anaconda3/lib/python3.12/site-packages/sumo"
)

import sumo_rl
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from rl_engine.rewards import congestion_aware_reward


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo_files"
NET_FILE = str(SUMO_DIR / "single_intersection.net.xml")
ROUTE_FILE = str(SUMO_DIR / "single_intersection.rou.xml")
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def _ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics collection callback
# ---------------------------------------------------------------------------
class MetricsCallback(BaseCallback):
    """Collects per-step metrics during training for live dashboards."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_metrics: list[dict] = []
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        self._ep_reward += self.locals.get("rewards", [0])[0]

        metric = {
            "timestep": self.num_timesteps,
            "reward": float(self.locals.get("rewards", [0])[0]),
        }
        # Pull SUMO-specific info if available
        for key in [
            "system_total_stopped",
            "system_total_waiting_time",
            "system_mean_waiting_time",
            "system_mean_speed",
        ]:
            if key in info:
                metric[key] = float(info[key])

        self.step_metrics.append(metric)

        # Check for episode end
        dones = self.locals.get("dones", [False])
        if any(dones):
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0

        return True


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
def make_env(
    fixed_ts: bool = False,
    num_seconds: int = 3600,
    delta_time: int = 5,
    use_gui: bool = False,
) -> sumo_rl.SumoEnvironment:
    """Create a single-agent SUMO-RL environment.

    Args:
        fixed_ts: If True, uses fixed-time signal control (baseline).
        num_seconds: Simulation length.
        delta_time: Decision interval in seconds.
        use_gui: Whether to launch SUMO-GUI.

    Returns:
        A Gymnasium-compatible environment.
    """
    env = sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        single_agent=True,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=3,
        min_green=5,
        max_green=60,
        fixed_ts=fixed_ts,
        reward_fn=congestion_aware_reward,
        sumo_seed=42,
        time_to_teleport=-1,
    )
    return env


# ---------------------------------------------------------------------------
# Run a fixed-time baseline
# ---------------------------------------------------------------------------
def run_fixed_time(
    num_seconds: int = 3600,
    progress_callback=None,
) -> dict[str, Any]:
    """Run the simulation with fixed-time signal control and collect metrics.

    Returns:
        Dictionary with aggregate metrics and per-step data.
    """
    _ensure_dirs()
    env = make_env(fixed_ts=True, num_seconds=num_seconds)

    obs, info = env.reset()
    done = False
    truncated = False
    step_data = []
    total_reward = 0.0
    step_count = 0

    while not done and not truncated:
        # Fixed-time controller: action is ignored, env handles signals
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        record = {
            "step": step_count,
            "sim_time": step_count * 5,
            "reward": float(reward),
            "total_reward": float(total_reward),
        }
        for key in [
            "system_total_stopped",
            "system_total_waiting_time",
            "system_mean_waiting_time",
            "system_mean_speed",
        ]:
            if key in info:
                record[key] = float(info[key])

        step_data.append(record)

        if progress_callback and step_count % 10 == 0:
            progress_callback(
                "fixed_time", step_count, num_seconds // 5, record
            )

    env.close()

    results = {
        "controller": "fixed_time",
        "total_reward": float(total_reward),
        "num_steps": step_count,
        "final_info": {k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
        "step_data": step_data,
    }

    # Save CSV
    csv_path = RESULTS_DIR / "fixed_time_results.csv"
    _save_csv(step_data, csv_path)

    return results


# ---------------------------------------------------------------------------
# Train RL agent
# ---------------------------------------------------------------------------
def train_rl(
    total_timesteps: int = 20000,
    num_seconds: int = 3600,
    progress_callback=None,
) -> dict[str, Any]:
    """Train a DQN agent on the traffic signal control task.

    Args:
        total_timesteps: Total training steps for DQN.
        num_seconds: Simulation duration per episode.
        progress_callback: Optional callback(controller, step, total, metrics).

    Returns:
        Dictionary with training results.
    """
    _ensure_dirs()
    env = make_env(fixed_ts=False, num_seconds=num_seconds)

    metrics_cb = MetricsCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=metrics_cb)
    train_time = time.time() - start_time

    # Save model
    model_path = str(MODELS_DIR / "dqn_traffic")
    model.save(model_path)

    env.close()

    results = {
        "controller": "rl_dqn",
        "total_timesteps": total_timesteps,
        "training_time_seconds": round(train_time, 2),
        "episode_rewards": [float(r) for r in metrics_cb.episode_rewards],
        "step_metrics": metrics_cb.step_metrics[-500:],  # Last 500 for dashboard
    }

    return results


# ---------------------------------------------------------------------------
# Evaluate trained RL agent
# ---------------------------------------------------------------------------
def evaluate_rl(
    num_seconds: int = 3600,
    progress_callback=None,
) -> dict[str, Any]:
    """Evaluate the trained DQN agent and collect metrics.

    Returns:
        Dictionary with evaluation metrics and per-step data.
    """
    _ensure_dirs()
    model_path = str(MODELS_DIR / "dqn_traffic")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"No trained model found at {model_path}.zip. Run training first."
        )

    env = make_env(fixed_ts=False, num_seconds=num_seconds)
    model = DQN.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    truncated = False
    step_data = []
    total_reward = 0.0
    step_count = 0

    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        record = {
            "step": step_count,
            "sim_time": step_count * 5,
            "reward": float(reward),
            "total_reward": float(total_reward),
        }
        for key in [
            "system_total_stopped",
            "system_total_waiting_time",
            "system_mean_waiting_time",
            "system_mean_speed",
        ]:
            if key in info:
                record[key] = float(info[key])

        step_data.append(record)

        if progress_callback and step_count % 10 == 0:
            progress_callback("rl_dqn", step_count, num_seconds // 5, record)

    env.close()

    results = {
        "controller": "rl_dqn",
        "total_reward": float(total_reward),
        "num_steps": step_count,
        "final_info": {k: float(v) for k, v in info.items() if isinstance(v, (int, float))},
        "step_data": step_data,
    }

    # Save CSV
    csv_path = RESULTS_DIR / "rl_dqn_results.csv"
    _save_csv(step_data, csv_path)

    return results


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------
def run_comparison(
    total_timesteps: int = 20000,
    num_seconds: int = 3600,
    progress_callback=None,
) -> dict[str, Any]:
    """Run full comparison: fixed-time baseline → RL training → RL evaluation.

    Returns:
        Dictionary with all results and comparison metrics.
    """
    _ensure_dirs()

    # Step 1: Fixed-time baseline
    fixed_results = run_fixed_time(num_seconds, progress_callback)

    # Step 2: Train RL
    train_results = train_rl(total_timesteps, num_seconds, progress_callback)

    # Step 3: Evaluate RL
    rl_results = evaluate_rl(num_seconds, progress_callback)

    # Comparison summary
    comparison = {
        "fixed_time": {
            "total_reward": fixed_results["total_reward"],
            "avg_reward_per_step": fixed_results["total_reward"] / max(fixed_results["num_steps"], 1),
            "final_metrics": fixed_results["final_info"],
        },
        "rl_dqn": {
            "total_reward": rl_results["total_reward"],
            "avg_reward_per_step": rl_results["total_reward"] / max(rl_results["num_steps"], 1),
            "final_metrics": rl_results["final_info"],
            "training_time": train_results["training_time_seconds"],
            "training_episodes": len(train_results["episode_rewards"]),
        },
        "improvement": {
            "reward_pct": round(
                (rl_results["total_reward"] - fixed_results["total_reward"])
                / abs(fixed_results["total_reward"])
                * 100,
                2,
            )
            if fixed_results["total_reward"] != 0
            else 0.0,
        },
    }

    # Save comparison JSON
    comp_path = RESULTS_DIR / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Save combined CSV
    _save_comparison_csv(fixed_results["step_data"], rl_results["step_data"])

    return {
        "fixed_time": fixed_results,
        "train": train_results,
        "rl_dqn": rl_results,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _save_csv(step_data: list[dict], path: Path):
    """Save step data to CSV."""
    if not step_data:
        return
    keys = step_data[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(step_data)


def _save_comparison_csv(fixed_data: list[dict], rl_data: list[dict]):
    """Save a combined comparison CSV."""
    path = RESULTS_DIR / "comparison.csv"
    rows = []

    for row in fixed_data:
        r = {"controller": "fixed_time", **row}
        rows.append(r)

    for row in rl_data:
        r = {"controller": "rl_dqn", **row}
        rows.append(r)

    if not rows:
        return

    keys = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
