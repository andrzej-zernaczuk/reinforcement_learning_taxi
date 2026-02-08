"""Visualization functions for comparing agents."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from reinforcement_learning_taxi.evaluation.metrics import \
    calculate_moving_average

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_learning_curves_comparison(
    rewards1: list[float],
    rewards2: list[float],
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    window: int = 100,
    title: str = "Learning Curves Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Compare learning curves of two agents.

    Args:
        rewards1: Episode rewards for agent 1
        rewards2: Episode rewards for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        window: Window size for moving average
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    episodes1 = range(1, len(rewards1) + 1)
    episodes2 = range(1, len(rewards2) + 1)

    moving_avg1 = calculate_moving_average(rewards1, window)
    moving_avg2 = calculate_moving_average(rewards2, window)

    ax.plot(episodes1, moving_avg1, linewidth=2, label=agent1_name, alpha=0.8)
    ax.plot(episodes2, moving_avg2, linewidth=2, label=agent2_name, alpha=0.8)

    ax.set_xlabel("Episode/Timestep", fontsize=12)
    ax.set_ylabel("Reward (Moving Average)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_performance_comparison(
    results1: dict,
    results2: dict,
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    title: str = "Performance Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Create bar chart comparing agent performance metrics.

    Args:
        results1: Evaluation results for agent 1
        results2: Evaluation results for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("mean_reward", "Average Reward"),
        ("success_rate", "Success Rate (%)"),
        ("mean_length", "Average Episode Length"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        values = [results1[metric], results2[metric]]
        if metric == "success_rate":
            values = [v * 100 for v in values]

        bars = ax.bar([agent1_name, agent2_name], values)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_success_rate_comparison(
    rewards1: list[float],
    rewards2: list[float],
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    window: int = 100,
    title: str = "Success Rate Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Compare success rates of two agents.

    Args:
        rewards1: Episode rewards for agent 1
        rewards2: Episode rewards for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        window: Window size for calculating success rate
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    def calculate_success_rates(rewards):
        success_rates = []
        episodes = []
        for i in range(window, len(rewards) + 1):
            window_rewards = rewards[i - window : i]
            rate = sum(1 for r in window_rewards if r > 0) / window * 100
            success_rates.append(rate)
            episodes.append(i)
        return episodes, success_rates

    episodes1, rates1 = calculate_success_rates(rewards1)
    episodes2, rates2 = calculate_success_rates(rewards2)

    ax.plot(episodes1, rates1, linewidth=2, label=agent1_name, alpha=0.8)
    ax.plot(episodes2, rates2, linewidth=2, label=agent2_name, alpha=0.8)
    ax.axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% Target")

    ax.set_xlabel("Episode/Timestep", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_training_efficiency(
    time1: float,
    time2: float,
    reward1: float,
    reward2: float,
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    title: str = "Training Efficiency",
    save_path: str | Path | None = None,
) -> None:
    """Compare training efficiency (time vs performance).

    Args:
        time1: Training time for agent 1
        time2: Training time for agent 2
        reward1: Final reward for agent 1
        reward2: Final reward for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    agents = [agent1_name, agent2_name]
    times = [time1 / 60, time2 / 60]
    rewards = [reward1, reward2]

    colors = sns.color_palette("husl", 2)

    for i, (agent, time, reward, color) in enumerate(zip(agents, times, rewards, colors)):
        ax.scatter(time, reward, s=200, alpha=0.7, color=color, label=agent)
        ax.annotate(
            agent,
            (time, reward),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
        )

    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_ylabel("Final Average Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_reward_distribution(
    rewards1: list[float],
    rewards2: list[float],
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    title: str = "Reward Distribution",
    save_path: str | Path | None = None,
) -> None:
    """Compare reward distributions of two agents.

    Args:
        rewards1: Episode rewards for agent 1
        rewards2: Episode rewards for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rewards1, bins=30, alpha=0.7, label=agent1_name, edgecolor="black")
    axes[0].axvline(np.mean(rewards1), color="r", linestyle="--", label=f"Mean: {np.mean(rewards1):.2f}")
    axes[0].set_xlabel("Reward", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title(f"{agent1_name} Reward Distribution", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(rewards2, bins=30, alpha=0.7, label=agent2_name, edgecolor="black", color="orange")
    axes[1].axvline(np.mean(rewards2), color="r", linestyle="--", label=f"Mean: {np.mean(rewards2):.2f}")
    axes[1].set_xlabel("Reward", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title(f"{agent2_name} Reward Distribution", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_comparison_summary(
    results1: dict,
    results2: dict,
    stats1: dict,
    stats2: dict,
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2",
    save_dir: str | Path | None = None,
) -> None:
    """Create comprehensive comparison summary plots.

    Args:
        results1: Evaluation results for agent 1
        results2: Evaluation results for agent 2
        stats1: Training statistics for agent 1
        stats2: Training statistics for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        save_dir: Directory to save plots (optional)
    """
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves_comparison(
        stats1["episode_rewards"],
        stats2["episode_rewards"],
        agent1_name,
        agent2_name,
        save_path=save_dir / "learning_curves_comparison.png" if save_dir else None,
    )

    plot_performance_comparison(
        results1,
        results2,
        agent1_name,
        agent2_name,
        save_path=save_dir / "performance_comparison.png" if save_dir else None,
    )

    plot_success_rate_comparison(
        stats1["episode_rewards"],
        stats2["episode_rewards"],
        agent1_name,
        agent2_name,
        save_path=save_dir / "success_rate_comparison.png" if save_dir else None,
    )

    if "training_time" in stats1 and "training_time" in stats2:
        plot_training_efficiency(
            stats1["training_time"],
            stats2["training_time"],
            results1["mean_reward"],
            results2["mean_reward"],
            agent1_name,
            agent2_name,
            save_path=save_dir / "training_efficiency.png" if save_dir else None,
        )
        )
