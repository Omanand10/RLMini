"""Deep Q-Network (Module 5) for PragyanEnv on randomized lunar maps.

Experience replay + target network + epsilon-greedy exploration + Huber loss.
The network sees only a 5x5 local NavCam window plus mission status, so it must
learn general driving skill that transfers to maps it has never seen.
"""
import time
import numpy as np
import torch
import torch.nn as nn

from .env import (PragyanEnv, FEATURE_DIM, N_ACTIONS, LunarMap, LANDING_ZONE, UNSAMPLED, HELD, SENT,
                  COMM_RADIUS)
from .tabular import frame


class QNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap):
        self.s = np.zeros((cap, FEATURE_DIM), np.float32)
        self.s2 = np.zeros((cap, FEATURE_DIM), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.m2 = np.zeros((cap, N_ACTIONS), bool)    # valid actions in s2
        self.g = np.zeros(cap, np.float32)             # bootstrap discount gamma^k
        self.cap, self.i, self.n = cap, 0, 0

    def add(self, s, a, r, s2, d, m2, g):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j], self.m2[j], self.g[j] = s, a, r, s2, d, m2, g
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, rng, k):
        idx = rng.integers(0, self.n, k)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]), torch.from_numpy(self.m2[idx]), torch.from_numpy(self.g[idx]))


def potential(m, state):
    """Phi(s) for potential-based shaping (Ng et al., 1999): mission progress minus the Chebyshev
    distance to the next goal (nearest unsampled site, else Vikram's radio range
    while data is held). F = gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged."""
    r, c, _, sites, _ = state
    todo = [(sr, sc) for (sr, sc, _), st in zip(m.sites, sites) if st == UNSAMPLED]
    if todo:
        d = min(max(abs(sr - r), abs(sc - c)) for sr, sc in todo)
    elif HELD in sites:
        d = max(0, max(abs(m.vikram[0] - r), abs(m.vikram[1] - c)) - COMM_RADIUS)
    else:
        d = 0
    # progress term (K > max distance) so Phi always rises when a sample is taken or sent
    progress = sum(1 for x in sites if x != UNSAMPLED) + sum(1 for x in sites if x == SENT)
    return PROGRESS_K * progress - float(d)


PROGRESS_K = 12.0

HPARAMS = dict(nstep=3, double=True, shaping=1.0, randomize="perturb", total_steps=600_000, lr=5e-4, gamma=0.99, batch=128, buffer=100_000,
               warmup=5_000, train_every=2, target_every=1_000,
               eps_start=1.0, eps_end=0.05, eps_frac=0.4, hidden=128)


def train_dqn(seed=0, record_at=(), log_every=20_000, **overrides):
    hp = {**HPARAMS, **overrides}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = PragyanEnv(randomize=hp["randomize"], obs_mode="features", seed=seed)
    q, tgt = QNet(hp["hidden"]), QNet(hp["hidden"])
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=hp["lr"])
    buf = Replay(hp["buffer"])
    eps_steps = int(hp["total_steps"] * hp["eps_frac"])

    log = {"returns": [], "steps": [], "success": [], "sent": [], "fail": [], "loss": [], "loss_step": []}
    recordings, record_at = {}, set(record_at)
    s, _ = env.reset()
    ep, G, t_ep, frames, losses, pend = 0, 0.0, 0, [], [], []
    t0 = time.time()
    for step in range(hp["total_steps"]):
        eps = max(hp["eps_end"], hp["eps_start"] - (hp["eps_start"] - hp["eps_end"]) * step / eps_steps)
        mask = env.action_mask()
        if rng.random() < eps:
            a = int(rng.choice(np.flatnonzero(mask)))
        else:
            with torch.no_grad():
                a = masked_argmax(q(torch.from_numpy(s)).numpy(), mask)
        phi = potential(env.map, env.state)
        s2, r, term, trunc, info = env.step(a)
        # shaped reward for learning only; the logged return G stays the true mission reward
        phi2 = 0.0 if term else potential(env.map, env.state)
        r_learn = r + hp["shaping"] * (hp["gamma"] * phi2 - phi)
        # n-step return: sum_{i<n} gamma^i r_{t+i} + gamma^n max_a Q(s_{t+n}, a)
        pend.append((s, a, r_learn))
        m2 = env.action_mask()
        if len(pend) == hp["nstep"] or term or trunc:
            while pend:
                R = sum(hp["gamma"] ** i * x[2] for i, x in enumerate(pend))
                buf.add(pend[0][0], pend[0][1], R, s2, float(term), m2, hp["gamma"] ** len(pend))
                pend.pop(0)
                if not (term or trunc):
                    break                     # slide the window by one step
        G += r; t_ep += 1
        if ep in record_at:
            frames.append(frame(env, a, r, info))
        s = s2

        if step >= hp["warmup"] and step % hp["train_every"] == 0:
            bs, ba, br, bs2, bd, bm2, bg = buf.sample(rng, hp["batch"])
            with torch.no_grad():
                if hp["double"]:     # Double DQN: online net picks a', target net scores it
                    a2 = q(bs2).masked_fill(~bm2, -1e9).argmax(1, keepdim=True)
                    q2 = tgt(bs2).gather(1, a2).squeeze(1)
                else:
                    q2 = tgt(bs2).masked_fill(~bm2, -1e9).max(1).values
                y = br + bg * (1 - bd) * q2
            pred = q(bs).gather(1, ba[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
        if step % hp["target_every"] == 0:
            tgt.load_state_dict(q.state_dict())

        if term or trunc:
            log["returns"].append(G); log["steps"].append(t_ep)
            log["success"].append(bool(info.get("success"))); log["sent"].append(int(info.get("sent", 0)))
            log["fail"].append(info["event"] if info["event"] in ("crater", "battery_dead", "night") else "")
            if ep in record_at:
                recordings[ep] = {"map": env.map.to_dict(), "frames": frames}
            ep += 1; G, t_ep, frames = 0.0, 0, []
            s, _ = env.reset()
        if losses and step % 1000 == 0:
            log["loss"].append(float(np.mean(losses))); log["loss_step"].append(step); losses = []
        if log_every and step % log_every == 0 and log["returns"]:
            print(f"  dqn step {step:>7} ep {ep:>6} eps {eps:.2f} "
                  f"ret {np.mean(log['returns'][-200:]):7.1f} succ {np.mean(log['success'][-200:]):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    log["train_time"] = time.time() - t0
    return q, log, recordings


def masked_argmax(qv, mask):
    return int(np.argmax(np.where(mask, qv, -1e9)))


def dqn_policy(q):
    from .env import features
    def act(env, rng):
        with torch.no_grad():
            return masked_argmax(q(torch.from_numpy(features(env.map, env.state))).numpy(), env.action_mask())
    return act


def export_weights(q):
    """Linear layer weights as nested lists for the in-browser forward pass."""
    layers = [m for m in q.net if isinstance(m, nn.Linear)]
    return [{"W": l.weight.detach().numpy().round(5).tolist(), "b": l.bias.detach().numpy().round(5).tolist()}
            for l in layers]


def eval_env_random(seed):
    return PragyanEnv(randomize="random", obs_mode="features", seed=seed)


def eval_env_perturb(seed):
    return PragyanEnv(randomize="perturb", obs_mode="features", seed=seed)


def eval_env_fixed(seed):
    return PragyanEnv(LunarMap.from_strings(LANDING_ZONE), obs_mode="features", seed=seed)
