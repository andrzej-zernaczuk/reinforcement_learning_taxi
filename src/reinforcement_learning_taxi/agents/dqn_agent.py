"""DQN agent wrapper for stable-baselines3."""

from pathlib import Path

from stable_baselines3 import DQN


class DQNAgent:
    """Wrapper for stable-baselines3 DQN agent.

    Provides consistent interface with TabularQLearningAgent.

    Args:
        env: Gymnasium environment
        policy: Policy type (e.g., 'MlpPolicy')
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        learning_starts: Number of steps before learning starts
        batch_size: Batch size for training
        gamma: Discount factor
        train_freq: Update frequency
        gradient_steps: Gradient steps per update
        target_update_interval: Target network update frequency
        exploration_fraction: Fraction of training for exploration
        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        policy_kwargs: Additional policy arguments
        verbose: Verbosity level
        seed: Random seed
    """

    def __init__(
        self,
        env,
        policy: str = "MlpPolicy",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 64,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        policy_kwargs: dict | None = None,
        verbose: int = 1,
        seed: int | None = None,
    ):
        self.model = DQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs or {},
            verbose=verbose,
            seed=seed,
        )

    def learn(self, total_timesteps: int, **kwargs) -> "DQNAgent":
        """Train the DQN agent.

        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional arguments for learn method

        Returns:
            Self
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def predict(self, state, deterministic: bool = True):
        """Predict action for given state.

        Args:
            state: Current state
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, state)
        """
        return self.model.predict(state, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        """Save agent to file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, env=None) -> "DQNAgent":
        """Load agent from file.

        Args:
            path: Path to saved agent
            env: Environment (optional)

        Returns:
            Loaded agent
        """
        agent = cls.__new__(cls)
        agent.model = DQN.load(path, env=env)
        return agent

    def set_parameters(self, params: dict) -> None:
        """Set model parameters.

        Args:
            params: Parameters dictionary
        """
        self.model.set_parameters(params)

    def get_parameters(self) -> dict:
        """Get model parameters.

        Returns:
            Parameters dictionary
        """
        return self.model.get_parameters()
