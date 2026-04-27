"""Helpers for converting between real-robot and simulator joint conventions."""

from __future__ import annotations

import numpy as np


BASE_JOINT_INDEX = 0
SHOULDER_JOINT_INDEX = 1


def wrap_joint_angles(joint_positions) -> np.ndarray:
    joint_positions = np.asarray(joint_positions, dtype=np.float64).reshape(3)
    return np.arctan2(np.sin(joint_positions), np.cos(joint_positions))


def real_to_sim_joint_positions(
    real_joint_positions,
    startup_real_joint_positions=None,
) -> np.ndarray:
    """Convert hardware joint readings into the simulator convention.

    When a startup pose is provided, the base joint is re-zeroed so the launch
    heading becomes the simulator's forward-facing reference.
    """

    sim_joint_positions = np.asarray(real_joint_positions, dtype=np.float64).reshape(3).copy()
    if startup_real_joint_positions is not None:
        startup_real_joint_positions = np.asarray(
            startup_real_joint_positions, dtype=np.float64
        ).reshape(3)
        sim_joint_positions[BASE_JOINT_INDEX] -= startup_real_joint_positions[BASE_JOINT_INDEX]

    sim_joint_positions[SHOULDER_JOINT_INDEX] *= -1
    return wrap_joint_angles(sim_joint_positions)


def sim_to_real_joint_positions(
    sim_joint_positions,
    startup_real_joint_positions=None,
) -> np.ndarray:
    """Convert simulator joint targets into the hardware convention."""

    real_joint_positions = np.asarray(sim_joint_positions, dtype=np.float64).reshape(3).copy()
    real_joint_positions[SHOULDER_JOINT_INDEX] *= -1
    if startup_real_joint_positions is not None:
        startup_real_joint_positions = np.asarray(
            startup_real_joint_positions, dtype=np.float64
        ).reshape(3)
        real_joint_positions[BASE_JOINT_INDEX] += startup_real_joint_positions[BASE_JOINT_INDEX]

    return wrap_joint_angles(real_joint_positions)
