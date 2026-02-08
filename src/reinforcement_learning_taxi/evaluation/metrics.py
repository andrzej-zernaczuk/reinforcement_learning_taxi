"""Evaluation metrics for RL agents."""

import inspect
import time
from typing import Any

import gymnasium as gym
import numpy as np
from scipy import stats

from reinforcement_learning_taxi.environments import (
    OneHotObservationWrapper,
    TaxiActionMaskWrapper,
    TaxiFeatureWrapper,
    TaxiRewardWrapper,
)


def _detect_reward_wrapper_class(env: gym.Env) -> type[TaxiRewardWrapper] | None:
    """Detect Taxi reward wrapper class used in environment stack."""
    current = env
    while hasattr(current, "env"):
        if isinstance(current, TaxiRewardWrapper):
            return type(current)
        current = current.env
    return None


def _detect_observation_wrapper(env: gym.Env) -> str | None:
    """Detect observation wrapper type used in environment stack."""
    current = env
    while hasattr(current, "env"):
        if isinstance(current, TaxiFeatureWrapper):
            return "feature"
        if isinstance(current, OneHotObservationWrapper):
            return "onehot"
        current = current.env
    return None


def _build_eval_env_like(env: gym.Env) -> gym.Env:
    """Create a fresh evaluation environment with wrapper parity."""
    # Create fresh evaluation environment to avoid state conflicts
    # with stable-baselines3 wrappers (DummyVecEnv, etc.)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "spec"):
        env_id = env.unwrapped.spec.id
    elif hasattr(env, "spec"):
        env_id = env.spec.id
    else:
        env_id = "Taxi-v3"

    eval_env = gym.make(env_id)

    # Preserve reward wrapper semantics when evaluating shaped environments.
    reward_wrapper_class = _detect_reward_wrapper_class(env)
    if reward_wrapper_class is not None:
        eval_env = reward_wrapper_class(eval_env)

    # Match observation wrapper used for training.
    obs_wrapper = _detect_observation_wrapper(env)
    if obs_wrapper == "feature":
        eval_env = TaxiFeatureWrapper(eval_env)
    elif obs_wrapper == "onehot":
        eval_env = OneHotObservationWrapper(eval_env)

    # Preserve action-mask wrapper when model uses MaskablePPO.
    if _uses_action_masking(env):
        eval_env = TaxiActionMaskWrapper(eval_env)

    return eval_env


def _uses_action_masking(env: gym.Env) -> bool:
    """Check whether the environment stack exposes action masks."""
    current = env
    while hasattr(current, "env"):
        if isinstance(current, TaxiActionMaskWrapper):
            return True
        current = current.env
    return hasattr(env, "action_masks")


def _predict_action(
    agent: Any,
    state: Any,
    deterministic: bool,
    env: gym.Env,
) -> int:
    """Predict next action, passing action masks when supported."""
    if not hasattr(agent, "predict"):
        return int(agent.select_action(state, training=not deterministic))

    kwargs: dict[str, Any] = {"deterministic": deterministic}

    try:
        signature = inspect.signature(agent.predict)
        if "action_masks" in signature.parameters and hasattr(env, "action_masks"):
            kwargs["action_masks"] = env.action_masks()
    except (TypeError, ValueError):
        pass

    action, _ = agent.predict(state, **kwargs)
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate agent performance over multiple episodes.

    Args:
        agent: Agent to evaluate (must have select_action method)
        env: Environment to evaluate in
        n_episodes: Number of evaluation episodes
        render: Whether to render environment
        deterministic: Use deterministic policy (no exploration)

    Returns:
        Dictionary with evaluation metrics
    """
    eval_env = _build_eval_env_like(env)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            if render:
                eval_env.render()

            action = _predict_action(
                agent=agent,
                state=state,
                deterministic=deterministic,
                env=eval_env,
            )

            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward > 0:
            successes += 1

    eval_env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": successes / n_episodes,
        "n_episodes": n_episodes,
    }


def calculate_success_rate(rewards: list[float]) -> float:
    """Calculate success rate from episode rewards.

    Args:
        rewards: List of episode rewards

    Returns:
        Success rate (0.0 to 1.0)
    """
    if not rewards:
        return 0.0
    successes = sum(1 for r in rewards if r > 0)
    return successes / len(rewards)


def calculate_average_reward(rewards: list[float]) -> float:
    """Calculate average reward.

    Args:
        rewards: List of episode rewards

    Returns:
        Average reward
    """
    return float(np.mean(rewards)) if rewards else 0.0


def calculate_convergence_speed(
    rewards: list[float],
    target_reward: float,
    window_size: int = 100,
) -> int | None:
    """Calculate episodes needed to reach target performance.

    Args:
        rewards: List of episode rewards
        target_reward: Target average reward
        window_size: Window for moving average

    Returns:
        Episode number when target was reached, or None if not reached
    """
    if len(rewards) < window_size:
        return None

    for i in range(window_size, len(rewards) + 1):
        window_avg = np.mean(rewards[i - window_size : i])
        if window_avg >= target_reward:
            return i

    return None


def calculate_moving_average(values: list[float], window: int = 100) -> list[float]:
    """Calculate moving average of values.

    Args:
        values: List of values
        window: Window size for moving average

    Returns:
        List of moving averages
    """
    if len(values) < window:
        return values

    moving_avg = []
    for i in range(len(values)):
        if i < window:
            moving_avg.append(np.mean(values[: i + 1]))
        else:
            moving_avg.append(np.mean(values[i - window + 1 : i + 1]))

    return moving_avg
