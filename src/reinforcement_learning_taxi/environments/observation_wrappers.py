"""Observation wrappers for Taxi-v3 environment."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class TaxiFeatureWrapper(gym.ObservationWrapper):
    """Convert Taxi-v3 discrete observation to feature vector.

    The Taxi-v3 environment returns a single integer (0-499) representing the state.
    This wrapper decodes it into semantic components and creates a compact feature vector
    using one-hot encoding for each component.

    The state is decoded as:
    - Taxi row (0-4): 5 features
    - Taxi column (0-4): 5 features
    - Passenger location (0-4, where 4=in taxi): 5 features
    - Destination (0-3): 4 features
    Total: 19 features

    Args:
        env: Gymnasium Taxi-v3 environment

    Raises:
        AssertionError: If the environment doesn't have a Discrete observation space
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, Discrete
        ), f"Expected Discrete observation space, got {type(env.observation_space)}"

        # Define new observation space: 19 binary features
        # 5 (taxi row) + 5 (taxi col) + 5 (passenger loc) + 4 (destination)
        self.observation_space = Box(low=0, high=1, shape=(19,), dtype=np.float32)

    def observation(self, obs: int) -> np.ndarray:
        """Convert discrete observation to feature vector.

        Args:
            obs: Discrete observation (integer 0-499)

        Returns:
            One-hot encoded feature vector of shape (19,)
        """
        # Decode the state into components
        taxi_row = obs // 100
        taxi_col = (obs // 20) % 5
        passenger_loc = (obs // 4) % 5
        destination = obs % 4

        # Create one-hot encoded features
        features = np.zeros(19, dtype=np.float32)

        # One-hot encode taxi row (indices 0-4)
        features[taxi_row] = 1.0

        # One-hot encode taxi column (indices 5-9)
        features[5 + taxi_col] = 1.0

        # One-hot encode passenger location (indices 10-14)
        features[10 + passenger_loc] = 1.0

        # One-hot encode destination (indices 15-18)
        features[15 + destination] = 1.0

        return features


class OneHotObservationWrapper(gym.ObservationWrapper):
    """Convert discrete observation to one-hot vector.

    This wrapper converts a discrete observation space into a one-hot encoded vector.
    For Taxi-v3 with 500 states, this creates a 500-dimensional binary vector.

    Note: This is less efficient than TaxiFeatureWrapper for Taxi-v3, but can be
    used for any discrete observation space.

    Args:
        env: Gymnasium environment with Discrete observation space

    Raises:
        AssertionError: If the environment doesn't have a Discrete observation space
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, Discrete
        ), f"Expected Discrete observation space, got {type(env.observation_space)}"

        self.n: int = env.observation_space.n
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def observation(self, obs: int) -> np.ndarray:
        """Convert discrete observation to one-hot vector.

        Args:
            obs: Discrete observation (integer)

        Returns:
            One-hot encoded vector of shape (n,)
        """
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot


class TaxiActionMaskWrapper(gym.Wrapper):
    """Expose valid-action masks for Taxi-v3.

    This wrapper provides an ``action_masks()`` method compatible with
    ``sb3-contrib`` MaskablePPO. Mask values are ``True`` for valid actions
    and ``False`` for invalid actions in the current state.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.action_space, Discrete
        ), f"Expected Discrete action space, got {type(env.action_space)}"

    def action_masks(self) -> np.ndarray:
        """Return current valid-action mask for Taxi-v3."""
        if hasattr(self.unwrapped, "action_mask") and hasattr(self.unwrapped, "s"):
            mask = self.unwrapped.action_mask(self.unwrapped.s)
            return np.asarray(mask, dtype=bool)

        # Fallback: derive legality from the transition table when available.
        if hasattr(self.unwrapped, "P") and hasattr(self.unwrapped, "s"):
            state = int(self.unwrapped.s)
            transitions = self.unwrapped.P.get(state, {})
            mask = np.zeros(self.action_space.n, dtype=bool)
            for action in range(self.action_space.n):
                for _, next_state, _, _ in transitions.get(action, []):
                    # Treat actions that keep the state unchanged as invalid.
                    if next_state != state:
                        mask[action] = True
                        break
            return mask

        # Last-resort fallback when no mask or transition table is available.
        return np.ones(self.action_space.n, dtype=bool)


def make_taxi_env(
    render_mode: str | None = None,
    use_feature_wrapper: bool = True,
    reward_wrapper_name: str | None = None,
    use_action_masking: bool = False,
) -> gym.Env:
    """Create Taxi-v3 environment with proper observation wrapper.

    Args:
        render_mode: Render mode for the environment
        use_feature_wrapper: Whether to use TaxiFeatureWrapper (recommended)
            If False, uses OneHotObservationWrapper instead
        reward_wrapper_name: Optional reward wrapper name from reward_wrappers.py
        use_action_masking: Whether to expose action masks for MaskablePPO

    Returns:
        Wrapped Gymnasium environment
    """
    from reinforcement_learning_taxi.environments.reward_wrappers import (  # type: ignore[import-untyped]
        get_reward_wrapper,
    )

    env = gym.make("Taxi-v3", render_mode=render_mode)

    # Apply reward wrapper first if specified
    if reward_wrapper_name:
        reward_wrapper_class = get_reward_wrapper(reward_wrapper_name)
        env = reward_wrapper_class(env)

    # Apply observation wrapper
    if use_feature_wrapper:
        env = TaxiFeatureWrapper(env)
    else:
        env = OneHotObservationWrapper(env)

    # Expose valid-action masks when using maskable algorithms.
    if use_action_masking:
        env = TaxiActionMaskWrapper(env)

    return env
