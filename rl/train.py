"""Train and evaluate every agent, then write results/run.pkl, results/metrics.json
and the site data (export_site_data.py). Plots are drawn by plots.py.

    python train.py            # full run (~15-20 min on 10 cores)
    python train.py --quick    # smoke test (~1 min)
"""
import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pragyan import env as E
from pragyan import tabular as T

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SEEDS = [10_000, 20_000, 30_000, 40_000, 50_000]
EVAL_EPISODES = 100


def evaluate_seeds(env_fn, policy, record=0):
    """Frozen-policy evaluation: EVAL_EPISODES episodes for each of 5 seeds."""
    per_seed, recs = [], []
    for k, seed in enumerate(EVAL_SEEDS):
        res, r = T.evaluate(env_fn(), policy, EVAL_EPISODES, seed=seed, record=record if k == 0 else 0)
        per_seed.append(res); recs += r
    keys = [k for k in per_seed[0] if k != "returns"]
    summary = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
    summary["return_std_seeds"] = float(np.std([p["return_mean"] for p in per_seed]))
    summary["success_std_seeds"] = float(np.std([p["success_rate"] for p in per_seed]))
    return summary, recs


def fixed_env():
    return E.PragyanEnv(seed=0)


# ---- worker jobs (top level so they can be pickled) ---------------------------------------

def tabular_job(job):
    name, algo, episodes, kw, main = job
    akw = dict(kw)
    if "eps_frac" in akw:          # schedules are lambdas, so build them inside the worker
        akw["eps"] = T.eps_schedule(episodes, frac=akw.pop("eps_frac"))
    env = fixed_env()
    rec_at = (0, episodes // 10, episodes // 2) if main else ()
    t0 = time.time()
    Q, log = getattr(T, algo)(env, episodes, record_at=rec_at, **akw)
    train_time = time.time() - t0
    summary, recs = evaluate_seeds(fixed_env, T.greedy_q_policy(Q), record=3 if main else 0)
    out = {"name": name, "algo": algo, "episodes": episodes, "kw": kw, "train_time": train_time,
           "log": {k: np.asarray(v) for k, v in log.as_dict().items()}, "eval": summary}
    if main:
        out["recordings"] = log.recordings
        out["final_episodes"] = recs
        out["Q"] = Q.astype(np.float32)
    return out


def dqn_job(job):
    total_steps, = job
    import torch
    torch.set_num_threads(8)
    from pragyan import dqn as D
    q, log, recs = D.train_dqn(total_steps=total_steps, record_at=(0,), log_every=max(1, total_steps // 20))
    pol = D.dqn_policy(q)
    evals, final_eps = {}, {}
    for env_name, fn in (("fixed", D.eval_env_fixed), ("perturbed", D.eval_env_perturb), ("random", D.eval_env_random)):
        evals[env_name], final_eps[env_name] = evaluate_seeds(lambda fn=fn: fn(0), pol, record=3)
    # perturbed-map episodes need their maps for the demo: replay with recording
    maps = []
    for ep in range(3):
        env = D.eval_env_perturb(0)
        env.reset(seed=EVAL_SEEDS[0] + ep)
        maps.append(env.map.to_dict())
    # (state, action) pairs on the fixed map for the in-browser network check
    checks = []
    for fr in final_eps["fixed"][0][:-1]:
        st = (fr[0], fr[1], fr[2], tuple(fr[3]), fr[4])
        envf = D.eval_env_fixed(0)
        envf.reset(); envf.state = st
        checks.append({"state": [fr[0], fr[1], fr[2], fr[3], fr[4]], "action": pol(envf, None),
                       "features": [round(float(x), 6) for x in E.features(envf.map, st)]})
    torch.save(q.state_dict(), os.path.join(ROOT, "results", "dqn.pt"))
    return {"name": "DQN", "log": {k: np.asarray(v) for k, v in log.items()}, "eval": evals,
            "final_episodes": final_eps, "perturbed_maps": maps, "first_episode": recs.get(0),
            "weights": D.export_weights(q), "checks": checks, "hparams": D.HPARAMS,
            "train_time": log["train_time"]}


def baseline_job(job):
    name, = job
    pol = {"Random": T.random_policy, "Heuristic": T.heuristic_policy}[name]
    from pragyan import dqn as D
    out = {"name": name}
    out["eval"], out["final_episodes"] = evaluate_seeds(fixed_env, pol, record=3)
    out["eval_perturbed"], out["final_episodes_perturbed"] = evaluate_seeds(lambda: D.eval_env_perturb(0), pol, record=3)
    out["eval_random"], _ = evaluate_seeds(lambda: D.eval_env_random(0), pol)
    return out


def vi_job(job):
    env = fixed_env()
    t0 = time.time()
    Q, info = T.value_iteration(env.map)
    info["plan_time"] = time.time() - t0
    summary, recs = evaluate_seeds(fixed_env, T.greedy_q_policy(Q), record=3)
    return {"name": "Value Iteration", "info": info, "eval": summary, "final_episodes": recs,
            "Q": Q.astype(np.float32)}


def run_job(job):
    kind, args = job
    return {"tabular": tabular_job, "dqn": dqn_job, "baseline": baseline_job, "vi": vi_job}[kind](args)


# ---- experiment plan -------------------------------------------------------------------------

def jobs(quick):
    N = 2_000 if quick else 300_000          # main tabular episodes
    S = 1_000 if quick else 100_000          # sweep episodes
    ND, SD = (400, 200) if quick else (60_000, 20_000)   # Dyna-Q (10x compute per step)
    main = [
        ("Q-learning", "q_learning", N, {}),
        ("SARSA", "sarsa", N, {}),
        ("n-step SARSA (n=4)", "n_step_sarsa", N, {"n": 4}),
        ("Monte Carlo", "mc_control", N, {}),
        ("Dyna-Q (k=10)", "dyna_q", ND, {"planning": 10}),
    ]
    out = [("dqn", (20_000 if quick else 2_000_000,)), ("vi", ())]
    out += [("tabular", (n, a, e, kw, True)) for n, a, e, kw in main]
    out += [("baseline", ("Random",)), ("baseline", ("Heuristic",))]
    sweeps = []
    for a in (0.05, 0.1, 0.2, 0.5):
        sweeps.append(("alpha", a, "q_learning", S, {"alpha": a}))
    for g in (0.9, 0.95, 0.99):
        sweeps.append(("gamma", g, "q_learning", S, {"gamma": g}))
    for f in (0.3, 0.5, 0.8):
        sweeps.append(("eps_decay", f, "q_learning", S, {"eps_frac": f}))
    for n in (1, 2, 4, 8):
        sweeps.append(("n", n, "n_step_sarsa", S, {"n": n}))
    for k in (0, 5, 10, 25):
        sweeps.append(("planning", k, "dyna_q", SD, {"planning": k}))
    out += [("tabular", (f"sweep:{p}={v}", algo, eps, kw, False)) for p, v, algo, eps, kw in sweeps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    todo = jobs(args.quick)
    t0 = time.time()
    results = []
    # phase 1: tabular agents, sweeps and baselines in parallel
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(run_job, [j for j in todo if j[0] != "dqn"]):
            print(f"[{time.time()-t0:6.0f}s] done: {res['name']}", flush=True)
            results.append(res)
    # phase 2: DQN alone, so the network gets every core instead of fighting the pool
    for job in [j for j in todo if j[0] == "dqn"]:
        results.append(run_job(job))
        print(f"[{time.time()-t0:6.0f}s] done: DQN", flush=True)

    run = {"quick": args.quick, "map": E.LunarMap.from_strings(E.LANDING_ZONE).to_dict(),
           "results": results, "total_time": time.time() - t0}
    with open(os.path.join(ROOT, "results", "run.pkl"), "wb") as f:
        pickle.dump(run, f)
    write_metrics(run)
    from export_site_data import export
    export(run)
    print(f"finished in {time.time()-t0:.0f}s")


# ---- outputs ---------------------------------------------------------------------------------

def by_name(run):
    return {r["name"]: r for r in run["results"]}


def write_metrics(run):
    R = by_name(run)
    table = {}
    for name, r in R.items():
        if name.startswith("sweep:"):
            continue
        ev = r["eval"]["fixed"] if name == "DQN" else r["eval"]
        row = {k: round(v, 4) for k, v in ev.items()}
        if "train_time" in r:
            row["train_time_s"] = round(r["train_time"], 1)
        if "episodes" in r:
            row["episodes"] = r["episodes"]
        table[name] = row
    sweeps = {n: {"return_mean": round(r["eval"]["return_mean"], 2), "success_rate": round(r["eval"]["success_rate"], 3),
                  "train_time_s": round(r["train_time"], 1)} for n, r in R.items() if n.startswith("sweep:")}
    d = R["DQN"]
    out = {"fixed_map": table, "sweeps": sweeps,
           "dqn_generalization": {"DQN": d["eval"],
                                  "Heuristic": {"fixed": R["Heuristic"]["eval"], "perturbed": R["Heuristic"]["eval_perturbed"],
                                                "random": R["Heuristic"]["eval_random"]},
                                  "Random": {"fixed": R["Random"]["eval"], "perturbed": R["Random"]["eval_perturbed"],
                                             "random": R["Random"]["eval_random"]}},
           "value_iteration": {k: v for k, v in R["Value Iteration"]["info"].items() if k != "deltas"},
           "dqn_hparams": d["hparams"], "total_time_s": round(run["total_time"], 1), "quick": run["quick"]}
    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
