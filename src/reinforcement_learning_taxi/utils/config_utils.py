"""Utilities for managing configuration files."""

from pathlib import Path

import yaml


def save_optimized_config(
    best_params: dict, base_config_path: str | Path, output_path: str | Path, algorithm: str = "PPO"
) -> None:
    """Save optimized configuration based on tuning results.

    Args:
        best_params: Dictionary with best hyperparameters from tuning
        base_config_path: Path to baseline configuration file
        output_path: Path to save optimized configuration
        algorithm: Algorithm name for documentation (PPO or DQN)
    """
    # Load base config
    base_config_path = Path(base_config_path)
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update agent parameters with best found values
    for param, value in best_params.items():
        if param in config["agent"]:
            if hasattr(value, "item"):
                value = value.item()
            config["agent"][param] = value

    # Update log directory to reflect optimization
    if "training" in config and "log_dir" in config["training"]:
        original_dir = config["training"]["log_dir"]
        config["training"]["log_dir"] = original_dir.replace("/logs/", "/logs_optimized/")

    # Add metadata about optimization
    config["_optimization_metadata"] = {
        "source": "hyperparameter_tuning",
        "baseline_config": str(base_config_path),
        "algorithm": algorithm,
        "optimized_parameters": list(best_params.keys()),
    }

    # Save optimized config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Optimized config saved to: {output_path}")


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
