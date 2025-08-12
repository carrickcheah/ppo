"""
Utility functions for loading YAML configuration files for the PPO scheduler.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


APP3_ROOT = Path("/Users/carrickcheah/Project/ppo/app3")
CONFIG_DIR = APP3_ROOT / "configs"


def load_yaml_config(filename: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a YAML config from app3/configs.

    Args:
        filename: YAML filename (e.g., "reward.yaml")
        default: Fallback dict if file missing or invalid

    Returns:
        Dict with config values (empty if not found and no default given)
    """
    cfg_path = CONFIG_DIR / filename
    if not cfg_path.exists():
        return dict(default) if default else {}

    try:
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return dict(default) if default else {}
            return data
    except Exception:
        return dict(default) if default else {}


def get_reward_config() -> Dict[str, Any]:
    """Load reward config and normalize keys to match RewardCalculator."""
    cfg = load_yaml_config(
        "reward.yaml",
        default={
            "on_time_reward": 100,
            "early_bonus_per_day": 50,
            "late_penalty_per_day": -100,
            "sequence_violation": -500,
            "utilization_bonus": 10,
            "action_taken_bonus": 5,
            "idle_penalty": -1,
        },
    )

    # Flatten nested 'rewards' section if present
    merged: Dict[str, Any] = {}
    merged.update(cfg)
    rewards_section = cfg.get("rewards")
    if isinstance(rewards_section, dict):
        merged.update(rewards_section)

    # Map alternative key names to RewardCalculator expected names
    if "sequence_violation" in merged and "sequence_violation_penalty" not in merged:
        merged["sequence_violation_penalty"] = merged.pop("sequence_violation")
    if "machine_utilization_bonus" in merged and "utilization_bonus" not in merged:
        merged["utilization_bonus"] = merged.pop("machine_utilization_bonus")

    # Keep only keys accepted by RewardCalculator
    allowed_keys = {
        "on_time_reward",
        "early_bonus_per_day",
        "late_penalty_per_day",
        "sequence_violation_penalty",
        "utilization_bonus",
        "action_taken_bonus",
        "idle_penalty",
    }
    filtered = {k: v for k, v in merged.items() if k in allowed_keys}

    return filtered


def get_environment_config() -> Dict[str, Any]:
    """Load environment configuration (planning horizon, steps, etc.)."""
    return load_yaml_config(
        "environment.yaml",
        default={
            "planning_horizon": 720,  # hours
            "time_step": 1,
            "max_steps_per_episode": 10000,
        },
    )


def get_data_config() -> Dict[str, Any]:
    """Load dataset paths configuration."""
    return load_yaml_config(
        "data.yaml",
        default={
            "stage_1_data": "data/10_jobs.json",
            "stage_2_data": "data/20_jobs.json",
            "stage_3_data": "data/40_jobs.json",
            "stage_4_data": "data/60_jobs.json",
            "stage_5_data": "data/100_jobs.json",
            "stage_6_data": "data/200_jobs.json",
        },
    )


def get_training_config() -> Dict[str, Any]:
    """Load training hyperparameters (not necessarily used by API)."""
    return load_yaml_config(
        "training.yaml",
        default={
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    )


