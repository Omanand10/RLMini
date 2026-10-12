# Project Pragyan: an RL mission planner for the Chandrayaan-3 lunar rover

This is an RL Lab CA mini project. A reinforcement-learning agent plans Pragyan's lunar day. It has to:
- Sample science sites with the right instrument (LIBS or APXS).
- Relay the data through the Vikram lander.
- Manage battery and sunlight.
- Avoid craters on slippery regolith.
- Hibernate safely before lunar night.

```
rllabca/
├── docs/     rubric, report template, verified mission facts
├── rl/       Python RL project: environment, agents, training, plots, results
├── report/   report generator + Pragyan_RL_Report.docx
└── site/     3D website + interactive mission simulator (Vite + React + three.js)
```

## RL project (`rl/`)
```bash
python3 -m venv .venv && .venv/bin/pip install -r rl/requirements.txt
cd rl
../.venv/bin/python -m pragyan.env          # environment self-check
../.venv/bin/python train.py                # train + evaluate every agent (~35 min on 10 cores)
../.venv/bin/python plots.py                # figures -> results/plots/
../.venv/bin/python export_site_data.py     # data for the website
```
- `pragyan/env.py` holds `PragyanEnv`, a Gymnasium environment. `outcomes()` is the single source of truth for the dynamics.
- `pragyan/tabular.py` holds Value Iteration, Monte Carlo control, Q-learning, SARSA, n-step SARSA, Dyna-Q and the baselines.
- `pragyan/dqn.py` holds the DQN: replay, target network, Double-DQN targets, n-step returns and action masking.

## Report (`report/`)
`../.venv/bin/python make_report.py` rebuilds `Pragyan_RL_Report.docx` in the LAB CA template format from `rl/results`.

## Website (`site/`)
```bash
cd site && npm install
npm run dev                          # development
npm run build && npm run preview     # offline presentation build
npm test                             # TypeScript simulator == Python environment check
```
