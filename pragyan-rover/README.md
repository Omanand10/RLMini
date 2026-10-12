# Project Pragyan: RL Mission Planner for the Chandrayaan-3 Lunar Rover

This project trains an RL agent to plan Pragyan's day on the Moon. The agent has to:
- Sample science sites with the correct instrument (LIBS or APXS).
- Relay the data through the Vikram lander.
- Manage battery and sunlight.
- Avoid craters on slippery regolith.
- Hibernate safely before lunar night.

The environment is inspired by the mission. It is not a replica of the real terrain.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pragyan.env          # environment self-check
.venv/bin/python train.py                # trains + evaluates every agent (~20-30 min, 10 cores)
.venv/bin/python plots.py                # figures -> results/plots/
open demo/index.html                     # simulation demo (works offline)
```
`train.py --quick` runs a 2-minute smoke test of the whole pipeline.

## Files
| Path | What |
|---|---|
| `pragyan/env.py` | `PragyanEnv` (Gymnasium) holds the MDP. `outcomes()` is the single source of truth for the dynamics. |
| `pragyan/tabular.py` | Value Iteration, Monte Carlo control, Q-learning, SARSA, n-step SARSA, Dyna-Q, and the baselines |
| `pragyan/dqn.py` | DQN with experience replay, a target network, Double-DQN targets and n-step returns |
| `train.py` | Experiments: main runs, hyper-parameter sweeps and evaluation. Writes `results/` and `demo/data.js`. |
| `plots.py` | Every graph used in the report |
| `demo/index.html` | Lunar simulation with three modes: Replay, Head-to-Head, and Live crater-drop (DQN runs in the browser) |

## Demo modes
- **Mission Replay:** watch any agent at episode 1, mid-training, or after training.
- **Head-to-Head:** two agents on the same map, side by side.
- **Live: Drop a Crater:** click to add craters or boulders while the DQN rover drives. It replans on every step, as Pragyan did at the 4 m crater on 27 Aug 2023.

The badge in the header confirms that the JavaScript environment reproduces every recorded Python transition and every DQN decision.
