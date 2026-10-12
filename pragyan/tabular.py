"""Tabular RL agents for PragyanEnv (fixed landing-zone map).

Module 2: Value Iteration (model-based optimal reference), Monte Carlo control
Module 3: Q-learning (off-policy TD), SARSA (on-policy TD)
Module 4: n-step SARSA, Dyna-Q (learning + planning with a learned model)
Baselines: random policy, greedy nearest-site heuristic
"""
import time
from collections import deque
import numpy as np

from .env import (N_ACTIONS, LIBS, APXS, TRANSMIT, RECHARGE, SLEEP, SHADOW, SITE_LIBS,
                  UNSAMPLED, HELD, SENT, MAX_BATTERY, DIRS, outcomes, state_index, n_states,
                  bfs_dist, valid_actions)

GAMMA = 0.99


# ---- shared helpers --------------------------------------------------------------------

def eps_schedule(episodes, eps0=1.0, eps_min=0.05, frac=0.8):
    """Exponential decay that reaches eps_min after `frac` of training."""
    decay = (eps_min / eps0) ** (1.0 / max(1, int(episodes * frac)))
    return lambda ep: max(eps_min, eps0 * decay ** ep)


NEG = -1e9   # Q value of a masked (invalid) action


def init_q(m):
    """Q table with invalid actions pre-masked to NEG, so every max/argmax
    (including bootstrapped targets and Dyna planning) only sees valid actions."""
    import itertools
    Q = np.zeros((n_states(m), N_ACTIONS))
    for r, c in itertools.product(range(m.H), range(m.W)):
        if not m.passable(r, c):
            continue
        for bat in range(MAX_BATTERY + 1):
            for sites in itertools.product(range(3), repeat=len(m.sites)):
                for ph in range(3):
                    st = (r, c, bat, sites, ph)
                    Q[state_index(m, st), ~valid_actions(m, st)] = NEG
    return Q


def egreedy(Q, s, eps, rng):
    q = Q[s]
    if rng.random() < eps:
        valid = np.flatnonzero(q > NEG / 2)
        return int(valid[rng.integers(len(valid))])
    best = np.flatnonzero(q == q.max())          # random tie-break
    return int(best[rng.integers(len(best))]) if len(best) > 1 else int(best[0])


def frame(env, a, rew, info):
    r, c, bat, sites, phase = env.state
    return [int(r), int(c), int(bat), list(sites), int(phase), int(a), float(rew), info["event"]]


class Logger:
    """Per-episode training metrics + recorded trajectories for the demo."""

    def __init__(self, record_at=()):
        self.returns, self.steps, self.success, self.sent, self.fail = [], [], [], [], []
        self.record_at = set(record_at)
        self.recordings = {}

    def end(self, ep, G, steps, info, frames):
        self.returns.append(G); self.steps.append(steps)
        self.success.append(bool(info.get("success")))
        self.sent.append(int(info.get("sent", 0)))
        ev = info.get("event")
        self.fail.append(ev if ev in ("crater", "battery_dead", "night") else "")
        if ep in self.record_at:
            self.recordings[ep] = frames

    def as_dict(self):
        return {"returns": self.returns, "steps": self.steps, "success": self.success,
                "sent": self.sent, "fail": self.fail}


# ---- Module 3: Q-learning and SARSA -------------------------------------------------------

def q_learning(env, episodes, alpha=0.1, gamma=GAMMA, eps=None, seed=0, record_at=()):
    rng = np.random.default_rng(seed)
    eps = eps or eps_schedule(episodes)
    Q = init_q(env.map)
    log = Logger(record_at)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1_000_003 + ep)
        G, frames, e = 0.0, [], eps(ep)
        for t in range(10_000):
            a = egreedy(Q, s, e, rng)
            s2, r, term, trunc, info = env.step(a)
            target = r if term else r + gamma * Q[s2].max()   # off-policy: max over next actions
            Q[s, a] += alpha * (target - Q[s, a])
            G += r
            if ep in log.record_at: frames.append(frame(env, a, r, info))
            s = s2
            if term or trunc: break
        log.end(ep, G, t + 1, info, frames)
    return Q, log


def sarsa(env, episodes, alpha=0.1, gamma=GAMMA, eps=None, seed=0, record_at=()):
    rng = np.random.default_rng(seed)
    eps = eps or eps_schedule(episodes)
    Q = init_q(env.map)
    log = Logger(record_at)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1_000_003 + ep)
        e = eps(ep)
        a = egreedy(Q, s, e, rng)
        G, frames = 0.0, []
        for t in range(10_000):
            s2, r, term, trunc, info = env.step(a)
            a2 = egreedy(Q, s2, e, rng)
            target = r if term else r + gamma * Q[s2, a2]     # on-policy: the action actually taken
            Q[s, a] += alpha * (target - Q[s, a])
            G += r
            if ep in log.record_at: frames.append(frame(env, a, r, info))
            s, a = s2, a2
            if term or trunc: break
        log.end(ep, G, t + 1, info, frames)
    return Q, log


# ---- Module 4: n-step SARSA and Dyna-Q -----------------------------------------------------

def n_step_sarsa(env, episodes, n=4, alpha=0.1, gamma=GAMMA, eps=None, seed=0, record_at=()):
    rng = np.random.default_rng(seed)
    eps = eps or eps_schedule(episodes)
    Q = init_q(env.map)
    log = Logger(record_at)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1_000_003 + ep)
        e = eps(ep)
        S, A, R = [s], [egreedy(Q, s, e, rng)], [0.0]
        T, t, G, frames = float("inf"), 0, 0.0, []
        while True:
            if t < T:
                s2, r, term, trunc, info = env.step(A[t])
                S.append(s2); R.append(r); G += r
                if ep in log.record_at: frames.append(frame(env, A[t], r, info))
                if term or trunc:
                    T = t + 1
                    boot_ok = not term          # truncated: bootstrap from last state
                else:
                    A.append(egreedy(Q, s2, e, rng))
            tau = t - n + 1
            if tau >= 0:
                ret = sum(gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, int(min(tau + n, T)) + 1))
                if tau + n < T:
                    ret += gamma ** n * Q[S[tau + n], A[tau + n]]
                elif T != float("inf") and boot_ok and tau + n >= T:
                    ret += gamma ** (T - tau) * Q[S[T]].max()
                Q[S[tau], A[tau]] += alpha * (ret - Q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        log.end(ep, G, int(T), info, frames)
    return Q, log


def dyna_q(env, episodes, planning=10, alpha=0.1, gamma=GAMMA, eps=None, seed=0, record_at=()):
    """Dyna-Q with a sample model: every observed (r, s', done) is stored per (s, a),
    so planning replays the empirical transition distribution of the stochastic Moon."""
    rng = np.random.default_rng(seed)
    eps = eps or eps_schedule(episodes)
    Q = init_q(env.map)
    model, seen = {}, []
    log = Logger(record_at)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1_000_003 + ep)
        G, frames, e = 0.0, [], eps(ep)
        for t in range(10_000):
            a = egreedy(Q, s, e, rng)
            s2, r, term, trunc, info = env.step(a)
            Q[s, a] += alpha * ((r if term else r + gamma * Q[s2].max()) - Q[s, a])
            key = (s, a)
            if key not in model:
                model[key] = deque(maxlen=32); seen.append(key)
            model[key].append((r, s2, term))
            for _ in range(planning):                              # imagined experience
                ps, pa = seen[rng.integers(len(seen))]
                exps = model[(ps, pa)]
                pr, ps2, pterm = exps[rng.integers(len(exps))]
                Q[ps, pa] += alpha * ((pr if pterm else pr + gamma * Q[ps2].max()) - Q[ps, pa])
            G += r
            if ep in log.record_at: frames.append(frame(env, a, r, info))
            s = s2
            if term or trunc: break
        log.end(ep, G, t + 1, info, frames)
    return Q, log


# ---- Module 2: Monte Carlo control and Value Iteration ---------------------------------------

def mc_control(env, episodes, alpha=0.05, gamma=GAMMA, eps=None, seed=0, record_at=()):
    """Every-visit on-policy Monte Carlo control with constant step size."""
    rng = np.random.default_rng(seed)
    eps = eps or eps_schedule(episodes)
    Q = init_q(env.map)
    log = Logger(record_at)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1_000_003 + ep)
        traj, frames, e = [], [], eps(ep)
        for t in range(10_000):
            a = egreedy(Q, s, e, rng)
            s2, r, term, trunc, info = env.step(a)
            traj.append((s, a, r))
            if ep in log.record_at: frames.append(frame(env, a, r, info))
            s = s2
            if term or trunc: break
        G = 0.0
        for s, a, r in reversed(traj):                    # returns computed backwards
            G = r + gamma * G
            Q[s, a] += alpha * (G - Q[s, a])
        log.end(ep, sum(x[2] for x in traj), len(traj), info, frames)
    return Q, log


def value_iteration(m, gamma=GAMMA, tol=1e-4, max_iter=2000):
    """Exact DP on the known model. Returns full-size Q table + sweep log."""
    from .env import MAX_BATTERY as MB
    start = (m.start[0], m.start[1], MB, (UNSAMPLED,) * len(m.sites), 0)
    idx = {start: 0}
    order = [start]
    rows, cols, probs = [], [], []
    R = []
    i = 0
    while i < len(order):                       # enumerate reachable states (BFS)
        st = order[i]
        for a in range(N_ACTIONS):
            er = 0.0
            for p, ns, rew, term, _ in outcomes(m, st, a):
                er += p * rew
                if not term:
                    j = idx.get(ns)
                    if j is None:
                        j = idx[ns] = len(order); order.append(ns)
                    rows.append(i * N_ACTIONS + a); cols.append(j); probs.append(p)
            R.append(er)
        i += 1
    S = len(order)
    rows, cols, probs = np.array(rows), np.array(cols), np.array(probs)
    R = np.array(R).reshape(S, N_ACTIONS)
    V = np.zeros(S)
    deltas = []
    for it in range(max_iter):
        Qc = R + gamma * np.bincount(rows, weights=probs * V[cols], minlength=S * N_ACTIONS).reshape(S, N_ACTIONS)
        V2 = Qc.max(1)
        deltas.append(float(np.abs(V2 - V).max()))
        V = V2
        if deltas[-1] < tol:
            break
    Q = np.zeros((n_states(m), N_ACTIONS))
    for st, j in idx.items():
        Q[state_index(m, st)] = Qc[j]
    return Q, {"n_reachable_states": S, "sweeps": len(deltas), "deltas": deltas, "V_start": float(V[0])}


# ---- policies + evaluation -------------------------------------------------------------------

def greedy_q_policy(Q):
    def act(env, rng):
        q = np.where(env.action_mask(), Q[state_index(env.map, env.state)], NEG)
        return int(np.argmax(q))
    return act


def random_policy(env, rng):
    return int(rng.choice(np.flatnonzero(env.action_mask())))


def heuristic_policy(env, rng):
    """Hand-written mission rule (what an engineer might script, no learning):
    recharge when low, sample when on a site, transmit when holding data in range,
    otherwise BFS to the nearest unsampled site (or back to Vikram), then sleep."""
    m = env.map
    r, c, bat, sites, phase = env.state
    in_sun = m.terrain[r, c] != SHADOW
    if bat <= 2 and in_sun:
        return RECHARGE
    i = m.site_at.get((r, c))
    if i is not None and sites[i] == UNSAMPLED and bat >= 2:
        return LIBS if m.sites[i][2] == SITE_LIBS else APXS
    held = any(s == HELD for s in sites)
    if held and m.in_range(r, c):
        return TRANSMIT
    todo = [(sr, sc) for (sr, sc, _), st in zip(m.sites, sites) if st == UNSAMPLED]
    if todo:
        goals = todo
    elif held:
        goals = [(rr, cc) for rr in range(m.H) for cc in range(m.W) if m.in_range(rr, cc)]
    else:
        if in_sun and bat >= 3 and phase == 2:
            return SLEEP
        return RECHARGE if in_sun else 0
    # move one step along a BFS shortest path to the closest goal
    best, best_d = None, 1e9
    for a, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if not m.passable(nr, nc) or m.terrain[nr, nc] == 3:
            continue
        dist = bfs_dist(m, (nr, nc))
        d = min(dist.get(g, 1e9) for g in goals)
        if d < best_d:
            best, best_d = a, d
    return best if best is not None else RECHARGE


def evaluate(env, policy, episodes=100, seed=10_000, record=0):
    """Run a frozen policy; returns summary + first `record` trajectories."""
    rng = np.random.default_rng(seed)
    G_all, steps, succ, sent, fail, recs = [], [], [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        G, frames = 0.0, []
        for t in range(10_000):
            a = policy(env, rng)
            _, r, term, trunc, info = env.step(a)
            G += r
            if ep < record: frames.append(frame(env, a, r, info))
            if term or trunc: break
        G_all.append(G); steps.append(t + 1); succ.append(bool(info.get("success")))
        sent.append(int(info.get("sent", 0)))
        fail.append(info["event"] if info["event"] in ("crater", "battery_dead", "night") else "")
        if ep < record: recs.append(frames)
    return {"return_mean": float(np.mean(G_all)), "return_std": float(np.std(G_all)),
            "success_rate": float(np.mean(succ)), "steps_mean": float(np.mean(steps)),
            "sent_mean": float(np.mean(sent)),
            "crater_rate": fail.count("crater") / episodes,
            "battery_rate": fail.count("battery_dead") / episodes,
            "night_rate": fail.count("night") / episodes,
            "returns": G_all}, recs


if __name__ == "__main__":
    from .env import PragyanEnv
    env = PragyanEnv(seed=0)
    t0 = time.time()
    Q, info = value_iteration(env.map)
    print("VI", {k: v for k, v in info.items() if k != "deltas"}, f"{time.time()-t0:.1f}s")
    res, _ = evaluate(env, greedy_q_policy(Q), 200)
    print("VI eval", {k: v for k, v in res.items() if k != "returns"})
    for name, pol in (("random", random_policy), ("heuristic", heuristic_policy)):
        res, _ = evaluate(env, pol, 200)
        print(name, {k: v for k, v in res.items() if k != "returns"})
    t0 = time.time()
    Q, log = q_learning(env, 2000)
    print("Q 2000 eps", f"{time.time()-t0:.1f}s", np.mean(log.returns[-200:]), np.mean(log.success[-200:]))
