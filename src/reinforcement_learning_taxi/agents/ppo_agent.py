"""PPO agent wrapper for stable-baselines3."""

from pathlib import Path

from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover - optional dependency
    MaskablePPO = None


class PPOAgent:
    """Wrapper for stable-baselines3 PPO agent.

    Provides consistent interface with DQNAgent.

    Args:
        env: Gymnasium environment
        policy: Policy type (e.g., 'MlpPolicy')
        learning_rate: Learning rate
        n_steps: Number of steps to run for each environment per update
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO
        clip_range_vf: Clipping parameter for value function
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: Maximum value for gradient clipping
        policy_kwargs: Additional policy arguments
        verbose: Verbosity level
        seed: Random seed
        use_action_masking: Use sb3-contrib MaskablePPO with action masks
    """

    def __init__(
        self,
        env,
        policy: str = "MlpPolicy",
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: dict | None = None,
        verbose: int = 1,
        seed: int | None = None,
        use_action_masking: bool = False,
    ):
        self.use_action_masking = use_action_masking

        if self.use_action_masking:
            if MaskablePPO is None:
                raise ImportError(
                    "Action masking requires sb3-contrib. "
                    "Install with: uv add sb3-contrib"
                )
            if not hasattr(env, "action_masks"):
                raise ValueError(
                    "Environment does not expose action masks. "
                    "Create it with make_taxi_env(use_action_masking=True)."
                )

        algo_class = MaskablePPO if self.use_action_masking else PPO

        self.model = algo_class(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs or {},
            verbose=verbose,
            seed=seed,
        )

    def learn(self, total_timesteps: int, **kwargs) -> "PPOAgent":
        """Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional arguments for learn method

        Returns:
            Self
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def predict(self, state, deterministic: bool = True, action_masks=None):
        """Predict action for given state.

        Args:
            state: Current state
            deterministic: Use deterministic policy
            action_masks: Optional valid-action mask for MaskablePPO

        Returns:
            Tuple of (action, state)
        """
        if self.use_action_masking:
            return self.model.predict(
                state,
                deterministic=deterministic,
                action_masks=action_masks,
            )
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
    def load(
        cls,
        path: str | Path,
        env=None,
        use_action_masking: bool | None = None,
    ) -> "PPOAgent":
        """Load agent from file.

        Args:
            path: Path to saved agent
            env: Environment (optional)
            use_action_masking: Force loader type, or auto-detect when None

        Returns:
            Loaded agent
        """
        agent = cls.__new__(cls)

        if use_action_masking is True:
            if MaskablePPO is None:
                raise ImportError(
                    "Action masking requires sb3-contrib. "
                    "Install with: uv add sb3-contrib"
                )
            agent.model = MaskablePPO.load(path, env=env)
            agent.use_action_masking = True
            return agent

        if use_action_masking is False:
            agent.model = PPO.load(path, env=env)
            agent.use_action_masking = False
            return agent

        try:
            agent.model = PPO.load(path, env=env)
            agent.use_action_masking = False
        except Exception:
            if MaskablePPO is None:
                raise
            agent.model = MaskablePPO.load(path, env=env)
            agent.use_action_masking = True

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
