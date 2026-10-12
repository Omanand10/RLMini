"""Export everything the website needs into site/src/data/pragyan.json.

Reads results/run.pkl + results/metrics.json (no retraining):
    python export_site_data.py
"""
import json
import os
import pickle

import numpy as np

from pragyan import env as E

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "site", "src", "data", "pragyan.json")
MAIN = ("Q-learning", "SARSA", "n-step SARSA (n=4)", "Monte Carlo", "Dyna-Q (k=10)")


def curve(y, points=300):
    """Moving-average smoothed curve, downsampled to ~points (x = episode number)."""
    y = np.asarray(y, dtype=float)
    w = max(50, len(y) // 100)
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[w:] - c[:-w]) / w
    x = np.arange(w, len(y) + 1)
    idx = np.unique(np.geomspace(1, len(s), points).astype(int) - 1)    # log-spaced: early learning visible
    return {"x": x[idx].tolist(), "y": np.round(s[idx], 3).tolist()}


def episodes(R):
    eps = {}

    def add(key, label, frames, m=None, agent="", stage=""):
        if frames:
            eps[key] = {"label": label, "agent": agent, "stage": stage, "frames": frames, "map": m}

    for name in MAIN:
        r = R[name]
        n = r["episodes"]
        for ep, stage, lab in ((0, "untrained", "Episode 1 (untrained)"),
                               (n // 10, "learning", f"Episode {n//10:,} (learning)"),
                               (n // 2, "improving", f"Episode {n//2:,} (improving)")):
            add(f"{name}|train{ep}", lab, r["recordings"].get(ep), agent=name, stage=stage)
        for i, fr in enumerate(r["final_episodes"]):
            add(f"{name}|final{i}", f"Test episode {i+1} (trained policy)", fr, agent=name, stage="trained")
    for name in ("Value Iteration", "Heuristic", "Random"):
        for i, fr in enumerate(R[name]["final_episodes"]):
            add(f"{name}|final{i}", f"Test episode {i+1}", fr, agent=name, stage="trained")
    d = R["DQN"]
    for i, fr in enumerate(d["final_episodes"]["fixed"]):
        add(f"DQN|final{i}", f"Test episode {i+1} (trained policy)", fr, agent="DQN", stage="trained")
    for i, fr in enumerate(d["final_episodes"]["perturbed"]):
        add(f"DQN|unseen{i}", f"Unseen hazard map {i+1}", fr, m=d["perturbed_maps"][i], agent="DQN", stage="unseen")
    for i, fr in enumerate(R["Heuristic"]["final_episodes_perturbed"]):
        add(f"Heuristic|unseen{i}", f"Unseen hazard map {i+1}", fr, m=d["perturbed_maps"][i], agent="Heuristic",
            stage="unseen")
    return eps


def export(run):
    R = {r["name"]: r for r in run["results"]}
    metrics = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
    consts = {k: getattr(E, k) for k in ("SLIP_P", "DRAIN_P", "DRAIN_P_SLOPE", "PHASE_P", "COMM_RADIUS", "MAX_BATTERY",
                                         "MAX_STEPS", "R_STEP", "R_SLOPE", "R_BUMP", "R_SAMPLE", "R_SENT", "R_WRONG",
                                         "R_CRATER", "R_DEAD", "R_SLEEP_SAFE", "R_SLEEP_UNSAFE", "R_NIGHT", "R_MISSION",
                                         "VIEW", "FEATURE_DIM")}
    curves = {name: {k: curve(R[name]["log"][k]) for k in ("returns", "success", "steps", "sent")} for name in MAIN}
    curves["DQN"] = {k: curve(R["DQN"]["log"][k]) for k in ("returns", "success", "steps", "sent")}
    d = R["DQN"]
    data = {"map": run["map"], "consts": consts, "actions": E.ACTION_NAMES, "episodes": episodes(R),
            "dqn": {"layers": d["weights"], "checks": d["checks"]},
            "table": metrics["fixed_map"], "generalization": metrics["dqn_generalization"],
            "sweeps": metrics["sweeps"], "valueIteration": metrics["value_iteration"], "curves": curves}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", os.path.relpath(OUT), f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    export(pickle.load(open(os.path.join(ROOT, "results", "run.pkl"), "rb")))
