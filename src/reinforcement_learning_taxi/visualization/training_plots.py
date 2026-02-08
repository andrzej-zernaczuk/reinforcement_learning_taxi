"""Visualization functions for training progress."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from reinforcement_learning_taxi.evaluation.metrics import calculate_moving_average

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_learning_curve(
    rewards: list[float],
    window: int = 100,
    title: str = "Learning Curve",
    save_path: str | Path | None = None,
) -> None:
    """Plot learning curve with moving average.

    Args:
        rewards: Episode rewards
        window: Window size for moving average
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = range(1, len(rewards) + 1)
    moving_avg = calculate_moving_average(rewards, window)

    ax.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    ax.plot(episodes, moving_avg, linewidth=2, label=f"Moving Average (window={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_success_rate(
    rewards: list[float],
    window: int = 100,
    title: str = "Success Rate Over Time",
    save_path: str | Path | None = None,
) -> None:
    """Plot success rate over time.

    Args:
        rewards: Episode rewards
        window: Window size for calculating success rate
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    success_rates = []
    episodes = []

    for i in range(window, len(rewards) + 1):
        window_rewards = rewards[i - window : i]
        success_rate = sum(1 for r in window_rewards if r > 0) / window
        success_rates.append(success_rate * 100)
        episodes.append(i)

    ax.plot(episodes, success_rates, linewidth=2)
    ax.axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% Target")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_epsilon_decay(
    epsilon_values: list[float],
    title: str = "Epsilon Decay",
    save_path: str | Path | None = None,
) -> None:
    """Plot epsilon decay over episodes.

    Args:
        epsilon_values: Epsilon values per episode
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = range(1, len(epsilon_values) + 1)
    ax.plot(episodes, epsilon_values, linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_episode_length(
    lengths: list[int],
    window: int = 100,
    title: str = "Episode Length Over Time",
    save_path: str | Path | None = None,
) -> None:
    """Plot episode length over time.

    Args:
        lengths: Episode lengths
        window: Window size for moving average
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = range(1, len(lengths) + 1)
    moving_avg = calculate_moving_average(lengths, window)

    ax.plot(episodes, lengths, alpha=0.3, label="Episode Length")
    ax.plot(episodes, moving_avg, linewidth=2, label=f"Moving Average (window={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_training_summary(
    stats: dict,
    save_dir: str | Path | None = None,
) -> None:
    """Create comprehensive training summary plots.

    Args:
        stats: Training statistics dictionary
        save_dir: Directory to save plots (optional)
    """
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(
        stats["episode_rewards"],
        title="Training Progress: Reward",
        save_path=save_dir / "learning_curve.png" if save_dir else None,
    )

    plot_success_rate(
        stats["episode_rewards"],
        title="Training Progress: Success Rate",
        save_path=save_dir / "success_rate.png" if save_dir else None,
    )

    if "epsilon_values" in stats:
        plot_epsilon_decay(
            stats["epsilon_values"],
            title="Exploration: Epsilon Decay",
            save_path=save_dir / "epsilon_decay.png" if save_dir else None,
        )

    if "episode_lengths" in stats:
        plot_episode_length(
            stats["episode_lengths"],
            title="Training Progress: Episode Length",
            save_path=save_dir / "episode_length.png" if save_dir else None,
        )
