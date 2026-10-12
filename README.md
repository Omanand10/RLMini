# Smart Traffic Control RL 🚦

A reinforcement learning application for adaptive traffic signal control using Eclipse SUMO, SUMO-RL, and Stable-Baselines3. Features a custom congestion-aware reward function, side-by-side fixed-time vs. RL comparative evaluation, a FastAPI backend with Server-Sent Events (SSE) streaming, and an interactive React + TypeScript + Tailwind dashboard.

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure Eclipse SUMO is installed on your machine and set `SUMO_HOME`:
```bash
# Example for conda / macOS:
export SUMO_HOME="/opt/anaconda3/lib/python3.12/site-packages/sumo"
# Or system-wide Homebrew/Linux install:
# export SUMO_HOME="/usr/local/share/sumo"
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install frontend dependencies:
```bash
cd dashboard
npm install
cd ..
```

---

## 🚀 Running the System

### Start the Backend (FastAPI + SUMO Simulation)
```bash
./start_backend.sh
```
* **API Server**: [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start the Frontend Dashboard (React + TypeScript + Tailwind)
In a separate terminal:
```bash
./start_dashboard.sh
```
* **Dashboard Web UI**: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Running Experiments from Command Line

You can also run benchmarks headlessly without launching the web server:

```bash
# Generate routes (optional, pre-generated file included)
python3 scripts/generate_routes.py --duration 3600

# Run full baseline vs DQN comparison
python3 scripts/run_experiment.py --timesteps 20000 --duration 3600
```

Results are saved to:
* `results/fixed_time_results.csv`
* `results/rl_dqn_results.csv`
* `results/comparison.csv`
* `results/comparison.json`
* `models/dqn_traffic.zip`

Detailed experimental findings and metrics are documented in [REPORT.md](REPORT.md).

---

## 📊 Dashboard Features

* **Simulation Controls**: One-click execution for Fixed-Time baseline, DQN RL Training, RL Evaluation, Full Comparative Pipeline, and State Reset.
* **Live Telemetry Charts**: Real-time streaming charts for cumulative rewards, average vehicle speeds, waiting times, and queue lengths.
* **Comparative Results Table**: Direct side-by-side percentage improvement breakdown between Fixed-Time and RL.
* **CSV & Report Export**: Direct in-browser download of raw simulation timeseries data.

---

## 🏗️ Architecture

```
RLMini/
├── backend/
│   ├── __init__.py
│   └── server.py              # FastAPI server + SSE streaming
├── dashboard/                 # React + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── App.tsx            # Main dashboard component & charts
│   │   ├── api.ts             # API client & SSE consumer
│   │   └── index.css          # Tailwind CSS design system
│   ├── package.json
│   └── vite.config.ts
├── models/                    # Saved RL policy checkpoints (.gitkeep)
├── results/                   # Simulation CSVs and JSON reports (.gitkeep)
├── rl_engine/
│   ├── __init__.py
│   ├── engine.py              # Training, baseline & evaluation runners
│   └── rewards.py             # Custom congestion-aware reward function
├── scripts/
│   ├── generate_routes.py     # Deterministic SUMO vehicle flow generator
│   └── run_experiment.py      # Standalone headless CLI experiment runner
├── sumo_files/                # SUMO network, edge, node, and configuration files
├── start_backend.sh           # Backend launcher script
├── start_dashboard.sh         # Frontend launcher script
├── requirements.txt           # Python dependencies
├── REPORT.md                  # Experimental results & benchmark analysis
└── .gitignore                 # GitHub ignore rules
```

---

## 📜 Attribution & License Notice

* **Eclipse SUMO**: Eclipse Public License Version 2.0 (EPL-2.0).
* **SUMO-RL**: Lucas N. Alegre (MIT License). Used as an installed Python library without local repository cloning.
