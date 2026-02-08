"""Trainer for DQN agent with stable-baselines3."""

import json
import time
from pathlib import Path

from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.monitor import Monitor

from reinforcement_learning_taxi.agents.dqn_agent import DQNAgent
from reinforcement_learning_taxi.evaluation.metrics import _build_eval_env_like
from reinforcement_learning_taxi.utils.path_utils import get_repo_root


class TrainingStatsCallback(BaseCallback):
    """Callback for collecting training statistics."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue

            self.episode_rewards.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))
            self.timesteps.append(self.num_timesteps)
        return True


class DQNTrainer:
    """Trainer for DQN agent using stable-baselines3.

    Args:
        env: Gymnasium environment
        agent: DQN agent to train
        log_dir: Directory to save logs
        checkpoint_freq: Save checkpoint every N timesteps
        eval_freq: Evaluate every N timesteps
        eval_episodes: Number of episodes for evaluation
    """

    def __init__(
        self,
        env,
        agent: DQNAgent,
        log_dir: str | Path = "results/logs/dqn",
        checkpoint_freq: int = 10000,
        eval_freq: int = 5000,
        eval_episodes: int = 100,
    ):
        self.env = env
        self.agent = agent
        log_dir = Path(log_dir)
        if not log_dir.is_absolute():
            log_dir = get_repo_root() / log_dir
        self.log_dir = log_dir
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "timesteps": [],
            "eval_rewards": [],
            "eval_success_rates": [],
            "eval_timesteps": [],
        }

    def train(self, total_timesteps: int) -> dict:
        """Train the DQN agent.

        Args:
            total_timesteps: Total training timesteps

        Returns:
            Training statistics dictionary
        """
        start_time = time.time()

        stats_callback = TrainingStatsCallback()

        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="dqn_model",
        )

        # Evaluation must run on an environment separate from training rollouts.
        eval_env = Monitor(_build_eval_env_like(self.env))

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.log_dir),
            log_path=str(self.log_dir),
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
        )

        callbacks = [stats_callback, checkpoint_callback, eval_callback]

        try:
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=100,
            )
        finally:
            eval_env.close()

        training_time = time.time() - start_time

        self.training_stats["episode_rewards"] = stats_callback.episode_rewards
        self.training_stats["episode_lengths"] = stats_callback.episode_lengths
        self.training_stats["timesteps"] = stats_callback.timesteps

        self.agent.save(self.log_dir / "final_model.zip")

        final_stats = {
            **self.training_stats,
            "training_time": training_time,
            "total_timesteps": total_timesteps,
        }

        self._save_stats(final_stats)
        self._save_summary(final_stats)

        return final_stats

    def _save_stats(self, stats: dict) -> None:
        """Save training statistics to JSON.

        Args:
            stats: Training statistics
        """
        stats_path = self.log_dir / "training_stats.json"

        serializable_stats = {
            k: [float(x) if hasattr(x, "item") else x for x in v] if isinstance(v, list) else v
            for k, v in stats.items()
        }

        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

    def _save_summary(self, stats: dict) -> None:
        """Save training summary.

        Args:
            stats: Training statistics
        """
        summary_path = self.log_dir / "summary.txt"

        with open(summary_path, "w") as f:
            f.write("DQN Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Timesteps: {stats['total_timesteps']}\n")
            f.write(f"Training Time: {stats['training_time']:.2f} seconds\n\n")

            if stats["episode_rewards"]:
                import numpy as np

                avg_reward = np.mean(stats["episode_rewards"][-100:])
                f.write(f"Final Average Reward (last 100 episodes): {avg_reward:.2f}\n")

                success_rate = sum(1 for r in stats["episode_rewards"][-100:] if r > 0) / min(
                    100, len(stats["episode_rewards"])
                )
                f.write(f"Final Success Rate: {success_rate:.2%}\n")

    @classmethod
    def load_stats(cls, log_dir: str | Path) -> dict:
        """Load training statistics from log directory.

        Args:
            log_dir: Directory containing saved statistics

        Returns:
            Training statistics dictionary
        """
        stats_path = Path(log_dir) / "training_stats.json"
        with open(stats_path, "r") as f:
            return json.load(f)
