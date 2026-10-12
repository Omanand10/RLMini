"""Custom congestion-aware reward function for SUMO-RL traffic signal control.

This module defines a reward function that penalizes congestion more
aggressively than the default waiting-time-based rewards, combining
queue length, lane density, and waiting time into a unified signal.
"""

from __future__ import annotations

import numpy as np


def congestion_aware_reward(traffic_signal) -> float:
    """Compute a congestion-aware reward for a single traffic signal.

    The reward combines three components:
    1. Queue penalty: penalizes total queued vehicles at the intersection.
    2. Density penalty: penalizes high lane densities (approaching gridlock).
    3. Waiting time penalty: penalizes accumulated waiting time.
    4. Speed bonus: rewards higher average speeds through the intersection.

    All components are normalized and combined with tunable weights.

    Args:
        traffic_signal: A sumo_rl TrafficSignal object.

    Returns:
        A scalar reward (higher is better).
    """
    # Queue component: total vehicles queued across all incoming lanes
    total_queued = traffic_signal.get_total_queued()
    queue_penalty = -0.25 * total_queued

    # Density component: average density across incoming lanes
    densities = traffic_signal.get_lanes_density()
    avg_density = np.mean(densities) if len(densities) > 0 else 0.0
    # Exponential penalty for high density (congestion is super-linear)
    density_penalty = -0.5 * (avg_density ** 2)

    # Waiting time component
    waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_waiting = sum(waiting_times) if waiting_times else 0.0
    waiting_penalty = -0.005 * total_waiting

    # Speed bonus: reward for maintaining flow
    avg_speed = traffic_signal.get_average_speed()
    speed_bonus = 0.3 * avg_speed

    reward = queue_penalty + density_penalty + waiting_penalty + speed_bonus
    return reward
