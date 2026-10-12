"""Standalone script to run the full experiment without the web UI.

Usage:
    export SUMO_HOME=/opt/anaconda3/lib/python3.12/site-packages/sumo
    python3 scripts/run_experiment.py [--timesteps 20000] [--duration 3600]
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault(
    "SUMO_HOME", "/opt/anaconda3/lib/python3.12/site-packages/sumo"
)


def main():
    parser = argparse.ArgumentParser(description="Run traffic control RL experiment")
    parser.add_argument("--timesteps", type=int, default=20000, help="RL training timesteps")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration (s)")
    args = parser.parse_args()

    from rl_engine.engine import run_comparison

    print("=" * 60)
    print("  Smart Traffic Control — RL Experiment")
    print("=" * 60)
    print(f"  Training timesteps: {args.timesteps}")
    print(f"  Simulation duration: {args.duration}s")
    print("=" * 60)
    print()

    results = run_comparison(
        total_timesteps=args.timesteps,
        num_seconds=args.duration,
    )

    comp = results["comparison"]

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Fixed-Time Total Reward:  {comp['fixed_time']['total_reward']:.2f}")
    print(f"  RL (DQN) Total Reward:    {comp['rl_dqn']['total_reward']:.2f}")
    print(f"  Improvement:              {comp['improvement']['reward_pct']:+.2f}%")
    print(f"  Training Time:            {comp['rl_dqn']['training_time']:.1f}s")
    print(f"  Training Episodes:        {comp['rl_dqn']['training_episodes']}")
    print()

    print("  Final SUMO Metrics:")
    for key in comp["fixed_time"]["final_metrics"]:
        fv = comp["fixed_time"]["final_metrics"].get(key, 0)
        rv = comp["rl_dqn"]["final_metrics"].get(key, 0)
        print(f"    {key:40s}  Fixed={fv:>8.2f}  RL={rv:>8.2f}")

    print()
    print("  Files saved:")
    print("    results/fixed_time_results.csv")
    print("    results/rl_dqn_results.csv")
    print("    results/comparison.csv")
    print("    results/comparison.json")
    print("    models/dqn_traffic.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
