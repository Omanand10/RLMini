"""Draw every figure for the report from results/run.pkl into results/plots/."""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from pragyan import env as E

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "plots")
MAIN = ["Q-learning", "SARSA", "n-step SARSA (n=4)", "Monte Carlo", "Dyna-Q (k=10)"]
COLORS = {"Q-learning": "#1f77b4", "SARSA": "#2ca02c", "n-step SARSA (n=4)": "#9467bd",
          "Monte Carlo": "#8c564b", "Dyna-Q (k=10)": "#d62728", "DQN": "#ff7f0e",
          "Value Iteration": "#222222", "Heuristic": "#7f7f7f", "Random": "#bcbd22"}
TERRAIN_RGB = {E.REGOLITH: "#b9b6b0", E.SLOPE: "#8f8a80", E.BOULDER: "#4a4540", E.CRATER: "#1c1a18",
               E.SHADOW: "#5d6470", E.VIKRAM: "#d4af37"}

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25})


def smooth(x, w):
    x = np.asarray(x, dtype=float)
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("  wrote", name)


def curves(R, key, ylabel, title, fname, scale=1.0, hlines=True):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name in MAIN:
        y = R[name]["log"][key].astype(float) * scale
        w = max(50, len(y) // 100)
        s = smooth(y, w)
        ax.plot(np.arange(w, len(y) + 1), s, label=name, color=COLORS[name], lw=1.4)
    if hlines and key == "returns":
        for ref in ("Value Iteration", "Heuristic", "Random"):
            ax.axhline(R[ref]["eval"]["return_mean"], ls="--", lw=1, color=COLORS[ref], label=f"{ref} (eval)")
    if hlines and key == "success":
        for ref in ("Value Iteration", "Heuristic"):
            ax.axhline(100 * R[ref]["eval"]["success_rate"], ls="--", lw=1, color=COLORS[ref], label=f"{ref} (eval)")
    ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    save(fig, fname)


def draw_map(ax, m, show_range=True):
    t = np.array(m["terrain"])
    H, W = t.shape
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c, r), 1, 1, color=TERRAIN_RGB[t[r, c]], lw=0))
            if t[r, c] == E.CRATER:
                ax.add_patch(Circle((c + .5, r + .5), .42, color="#0a0908"))
    vr, vc = m["vikram"]
    if show_range:
        R = E.COMM_RADIUS
        ax.add_patch(Rectangle((vc - R, vr - R), 2 * R + 1, 2 * R + 1, fill=False, ls="--", ec="#d4af37", lw=1.2))
    ax.text(vc + .5, vr + .55, "V", ha="center", va="center", fontsize=9, weight="bold", color="black")
    for r, c, typ in m["sites"]:
        ax.add_patch(Circle((c + .5, r + .5), .3, color="#e74c3c" if typ == E.SITE_LIBS else "#3498db"))
        ax.text(c + .5, r + .52, "L" if typ == E.SITE_LIBS else "A", ha="center", va="center", fontsize=7,
                color="white", weight="bold")
    sr, sc = m["start"]
    ax.plot(sc + .5, sr + .5, marker="s", color="#2ecc71", ms=6)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def path_xy(m, frames):
    xs, ys = [m["start"][1] + .5], [m["start"][0] + .5]
    for f in frames:
        xs.append(f[1] + .5); ys.append(f[0] + .5)
    return xs, ys


def fig_environment(run):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    draw_map(ax, run["map"])
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, color=TERRAIN_RGB[k]) for k in (E.REGOLITH, E.SLOPE, E.BOULDER, E.SHADOW)]
    handles += [Circle((0, 0), 1, color="#0a0908"), Circle((0, 0), 1, color="#e74c3c"), Circle((0, 0), 1, color="#3498db"),
                Line2D([], [], marker="s", color="#2ecc71", ls=""), Line2D([], [], ls="--", color="#d4af37")]
    labels = ["Regolith", "Slope", "Boulder", "Permanent shadow", "Crater", "LIBS site", "APXS site",
              "Rover start", "Vikram radio range"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title("Lunar landing-zone environment (12 x 12 grid, 1 cell ~ 1 m)")
    save(fig, "environment_map.png")


def fig_final_comparison(R):
    names = ["Random", "Heuristic"] + MAIN + ["DQN", "Value Iteration"]
    ev = {n: (R[n]["eval"]["fixed"] if n == "DQN" else R[n]["eval"]) for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    x = np.arange(len(names))
    cols = [COLORS[n] for n in names]
    axes[0].bar(x, [ev[n]["return_mean"] for n in names], yerr=[ev[n]["return_std_seeds"] for n in names],
                color=cols, capsize=3)
    axes[0].set_title("Average return per mission (500 eval episodes)")
    axes[1].bar(x, [100 * ev[n]["success_rate"] for n in names], color=cols,
                yerr=[100 * ev[n]["success_std_seeds"] for n in names], capsize=3)
    axes[1].set_title("Mission success rate (%)")
    bottom = np.zeros(len(names))
    for key, lab, col in (("crater_rate", "Fell in crater", "#222"), ("battery_rate", "Battery dead", "#e67e22"),
                          ("night_rate", "Caught by night", "#34495e")):
        v = np.array([100 * ev[n][key] for n in names])
        axes[2].bar(x, v, bottom=bottom, label=lab, color=col); bottom += v
    axes[2].set_title("Failure modes (% of missions)"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    save(fig, "final_comparison.png")


def fig_value_policy(R, run):
    m = E.LunarMap.from_strings(E.LANDING_ZONE)
    arrows = {i: (dc * .32, dr * .32) for i, (dr, dc) in enumerate(E.DIRS)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, name in zip(axes, ("Value Iteration", "Q-learning")):
        Q = R[name]["Q"]
        V = np.full((m.H, m.W), np.nan)
        draw_map(ax, run["map"])
        for r in range(m.H):
            for c in range(m.W):
                if not m.passable(r, c) or m.terrain[r, c] == E.CRATER:
                    continue
                st = (r, c, E.MAX_BATTERY, (0, 0, 0, 0), 0)
                q = np.where(E.valid_actions(m, st), Q[E.state_index(m, st)], -1e9)
                V[r, c] = q.max()
                a = int(q.argmax())
                if a < 8:
                    dx, dy = arrows[a]
                    ax.annotate("", (c + .5 + dx, r + .5 + dy), (c + .5 - dx, r + .5 - dy),
                                arrowprops=dict(arrowstyle="->", color="white", lw=1))
                else:
                    ax.text(c + .5, r + .5, E.ACTION_NAMES[a][:2], ha="center", va="center", fontsize=6, color="white")
        im = ax.imshow(np.ma.masked_invalid(V), extent=(0, m.W, m.H, 0), cmap="viridis", alpha=.6, zorder=3)
        ax.set_title(f"{name}: V(s) and greedy action\n(battery full, nothing sampled, morning)")
        plt.colorbar(im, ax=ax, fraction=.046)
    save(fig, "value_policy_heatmap.png")


def fig_paths(R, run):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, name in zip(axes, ("Q-learning", "SARSA", "Heuristic")):
        draw_map(ax, run["map"])
        for i, fr in enumerate(R[name]["final_episodes"]):
            xs, ys = path_xy(run["map"], fr)
            ax.plot(xs, ys, color=["#00e5ff", "#ffea00", "#ff4081"][i], lw=2, alpha=.85,
                    label=f"run {i+1}: {fr[-1][7]}")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(f"{name} - greedy trajectories")
    save(fig, "trajectories_q_vs_sarsa.png")


def fig_sweeps(R):
    groups = [("alpha", "Q-learning: learning rate alpha"), ("gamma", "Q-learning: discount gamma"),
              ("eps_decay", "Q-learning: epsilon decay length (fraction)"), ("n", "n-step SARSA: n"),
              ("planning", "Dyna-Q: planning steps k")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (p, title) in zip(axes.flat, groups):
        runs = sorted([r for n, r in R.items() if n.startswith(f"sweep:{p}=")], key=lambda r: float(r["name"].split("=")[1]))
        cmap = plt.cm.viridis(np.linspace(0, .9, len(runs)))
        for col, r in zip(cmap, runs):
            y = r["log"]["returns"].astype(float)
            w = max(50, len(y) // 100)
            ax.plot(np.arange(w, len(y) + 1), smooth(y, w), color=col, lw=1.3,
                    label=f"{p}={r['name'].split('=')[1]}  eval {r['eval']['return_mean']:.0f}")
        ax.set_title(title); ax.set_xlabel("Episode"); ax.set_ylabel("Return (smoothed)")
        ax.legend(fontsize=7)
    # sample efficiency vs compute for Dyna-Q
    ax = axes.flat[5]
    runs = sorted([r for n, r in R.items() if n.startswith("sweep:planning=")], key=lambda r: float(r["name"].split("=")[1]))
    ks = [int(r["name"].split("=")[1]) for r in runs]
    ax.bar([str(k) for k in ks], [r["train_time"] for r in runs], color="#d62728", alpha=.7)
    ax.set_title("Dyna-Q: wall-clock training time"); ax.set_xlabel("planning steps k"); ax.set_ylabel("seconds")
    save(fig, "hyperparameter_sweeps.png")


def fig_dqn(R):
    d = R["DQN"]
    log = d["log"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    y = log["returns"].astype(float); w = max(50, len(y) // 100)
    axes[0].plot(np.arange(w, len(y) + 1), smooth(y, w), color=COLORS["DQN"])
    axes[0].set_title("DQN: return vs episode (hazard-randomised maps)"); axes[0].set_xlabel("Episode")
    s = log["success"].astype(float) * 100
    axes[1].plot(np.arange(w, len(s) + 1), smooth(s, w), color=COLORS["DQN"])
    axes[1].set_title("DQN: success rate vs episode (%)"); axes[1].set_xlabel("Episode")
    axes[2].plot(log["loss_step"], log["loss"], color="#555")
    axes[2].set_yscale("log"); axes[2].set_title("DQN: Huber TD loss"); axes[2].set_xlabel("Environment step")
    save(fig, "dqn_training.png")

    envs = ["fixed", "perturbed", "random"]
    labels = ["Training landing zone", "Unseen hazards\n(perturbed zone)", "Fully random\nlunar maps"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(3); wbar = .27
    for i, (name, get) in enumerate((("DQN", lambda e: d["eval"][e]),
                                     ("Heuristic", lambda e: R["Heuristic"]["eval" if e == "fixed" else f"eval_{e}"]),
                                     ("Random", lambda e: R["Random"]["eval" if e == "fixed" else f"eval_{e}"]))):
        axes[0].bar(x + (i - 1) * wbar, [get(e)["return_mean"] for e in envs], wbar, label=name, color=COLORS[name])
        axes[1].bar(x + (i - 1) * wbar, [100 * get(e)["success_rate"] for e in envs], wbar, label=name, color=COLORS[name])
    for ax, t in zip(axes, ("Average return", "Success rate (%)")):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title(f"Generalisation: {t}"); ax.legend(fontsize=8)
    save(fig, "dqn_generalization.png")


def fig_convergence(R):
    names = MAIN
    eps90, times = [], []
    for n in names:
        y = smooth(R[n]["log"]["returns"].astype(float), max(50, len(R[n]["log"]["returns"]) // 100))
        lo, hi = y[0], y[-1]
        idx = np.argmax(y >= lo + .9 * (hi - lo))
        eps90.append(idx); times.append(R[n]["train_time"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cols = [COLORS[n] for n in names]
    axes[0].bar(names, eps90, color=cols); axes[0].set_title("Episodes to reach 90% of final performance")
    axes[1].bar(names, times, color=cols); axes[1].set_title("Wall-clock training time (s)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    save(fig, "convergence_speed.png")


def fig_vi(R):
    d = R["Value Iteration"]["info"]["deltas"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(np.arange(1, len(d) + 1), d, marker=".", color="#222")
    ax.set_xlabel("Sweep"); ax.set_ylabel(r"max |V$_{k+1}$ - V$_k$|")
    ax.set_title(f"Value Iteration convergence ({R['Value Iteration']['info']['n_reachable_states']:,} reachable states)")
    save(fig, "value_iteration_convergence.png")


def fig_sent(R):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name in MAIN:
        y = R[name]["log"]["sent"].astype(float)
        w = max(50, len(y) // 100)
        ax.plot(np.arange(w, len(y) + 1), smooth(y, w), label=name, color=COLORS[name], lw=1.4)
    ax.set_xscale("log"); ax.set_xlabel("Episode"); ax.set_ylabel("Samples relayed to Vikram (of 4)")
    ax.set_title("Science return vs episode"); ax.legend(fontsize=8)
    save(fig, "samples_sent_vs_episode.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    run = pickle.load(open(os.path.join(ROOT, "results", "run.pkl"), "rb"))
    R = {r["name"]: r for r in run["results"]}
    fig_environment(run)
    curves(R, "returns", "Return (smoothed)", "Reward vs Episode (fixed landing zone)", "reward_vs_episode.png")
    curves(R, "success", "Success rate (%)", "Success Rate vs Episode", "success_vs_episode.png", scale=100)
    curves(R, "steps", "Steps per episode", "Steps/Episode vs Episode", "steps_vs_episode.png")
    fig_sent(R)
    fig_final_comparison(R)
    fig_value_policy(R, run)
    fig_paths(R, run)
    fig_sweeps(R)
    fig_dqn(R)
    fig_convergence(R)
    fig_vi(R)


if __name__ == "__main__":
    main()
