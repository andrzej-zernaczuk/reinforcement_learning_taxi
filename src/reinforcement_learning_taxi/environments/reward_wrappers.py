"""Custom reward wrappers for Taxi-v3 environment."""

import gymnasium as gym
import numpy as np


class TaxiRewardWrapper(gym.Wrapper):
    """Base class for Taxi reward wrappers.

    All custom reward functions inherit from this class and override the reward() method.

    Args:
        env: Gymnasium Taxi environment
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_state = None
        self.passenger_location = None
        self.destination = None

    def _decode_state(self, state: int) -> tuple[int, int, int, int]:
        """Decode state into components.

        Args:
            state: Encoded state integer

        Returns:
            Tuple of (taxi_row, taxi_col, passenger_loc, destination)
        """
        taxi_row = state // 100
        taxi_col = (state // 20) % 5
        passenger_loc = (state // 4) % 5
        destination = state % 4
        return taxi_row, taxi_col, passenger_loc, destination

    def _manhattan_distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Calculate Manhattan distance between two positions.

        Args:
            row1: Row of first position
            col1: Column of first position
            row2: Row of second position
            col2: Column of second position

        Returns:
            Manhattan distance
        """
        return abs(row1 - row2) + abs(col1 - col2)

    def _get_location_coords(self, location: int) -> tuple[int, int]:
        """Get coordinates of passenger pickup/dropoff locations.

        Args:
            location: Location index (0=R, 1=G, 2=Y, 3=B)

        Returns:
            Tuple of (row, col)
        """
        locations = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3),  # B
        }
        return locations.get(location, (0, 0))

    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)
        self.last_state = obs
        return obs, info

    def step(self, action):
        """Execute action and apply custom reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply custom reward modification
        custom_reward = self.modify_reward(self.last_state, action, reward, obs, terminated)

        self.last_state = obs
        return obs, custom_reward, terminated, truncated, info

    def modify_reward(
        self, state: int, action: int, original_reward: float, next_state: int, done: bool
    ) -> float:
        """Modify reward - to be overridden by subclasses.

        Args:
            state: Current state
            action: Action taken
            original_reward: Original environment reward
            next_state: Next state
            done: Whether episode is done

        Returns:
            Modified reward
        """
        return original_reward


class DefaultReward(TaxiRewardWrapper):
    """Default Taxi-v3 reward (no modification).

    Rewards:
    - -1 per step
    - +20 for successful delivery
    - -10 for illegal pickup/dropoff
    """

    def modify_reward(
        self, state: int, action: int, original_reward: float, next_state: int, done: bool
    ) -> float:
        """Keep original reward unchanged."""
        return original_reward


class DistanceBasedReward(TaxiRewardWrapper):
    """Distance-based reward shaping.

    Adds small rewards for moving closer to the passenger (when not picked up)
    or closer to destination (when passenger is in taxi).

    Rewards:
    - Original rewards remain
    - +0.1 for moving closer to target
    - -0.1 for moving away from target
    """

    def modify_reward(
        self, state: int, action: int, original_reward: float, next_state: int, done: bool
    ) -> float:
        """Add distance-based shaping."""
        # Decode states
        taxi_row, taxi_col, pass_loc, dest = self._decode_state(state)
        next_taxi_row, next_taxi_col, next_pass_loc, next_dest = self._decode_state(next_state)

        shaping_reward = 0.0

        # If passenger not in taxi (pass_loc < 4), reward getting closer to passenger
        if pass_loc < 4:
            pass_row, pass_col = self._get_location_coords(pass_loc)
            old_dist = self._manhattan_distance(taxi_row, taxi_col, pass_row, pass_col)
            new_dist = self._manhattan_distance(next_taxi_row, next_taxi_col, pass_row, pass_col)

            if new_dist < old_dist:
                shaping_reward += 0.1  # Moved closer to passenger
            elif new_dist > old_dist:
                shaping_reward -= 0.1  # Moved away from passenger

        # If passenger in taxi (pass_loc == 4), reward getting closer to destination
        elif pass_loc == 4:
            dest_row, dest_col = self._get_location_coords(dest)
            old_dist = self._manhattan_distance(taxi_row, taxi_col, dest_row, dest_col)
            new_dist = self._manhattan_distance(next_taxi_row, next_taxi_col, dest_row, dest_col)

            if new_dist < old_dist:
                shaping_reward += 0.1  # Moved closer to destination
            elif new_dist > old_dist:
                shaping_reward -= 0.1  # Moved away from destination

        return original_reward + shaping_reward


class ModifiedPenaltyReward(TaxiRewardWrapper):
    """Modified step penalty reward.

    Reduces the per-step penalty to encourage exploration.

    Rewards:
    - -0.5 per step (instead of -1)
    - +20 for successful delivery
    - -10 for illegal pickup/dropoff
    """

    def modify_reward(
        self, state: int, action: int, original_reward: float, next_state: int, done: bool
    ) -> float:
        """Reduce step penalty."""
        # If it's a step penalty (-1), make it less harsh
        if original_reward == -1:
            return -0.5
        return original_reward


class EnhancedReward(TaxiRewardWrapper):
    """Enhanced reward with multiple modifications.

    Combines distance-based shaping with pickup/dropoff bonuses.

    Rewards:
    - -1 per step (original)
    - +25 for successful delivery (increased from +20)
    - -5 for illegal pickup/dropoff (reduced from -10)
    - +0.2 for moving closer to target
    - -0.2 for moving away from target
    - +2 for successful pickup
    """

    def modify_reward(
        self, state: int, action: int, original_reward: float, next_state: int, done: bool
    ) -> float:
        """Apply enhanced reward modifications."""
        modified_reward = original_reward

        # Decode states
        taxi_row, taxi_col, pass_loc, dest = self._decode_state(state)
        next_taxi_row, next_taxi_col, next_pass_loc, next_dest = self._decode_state(next_state)

        # Modify success reward
        if original_reward == 20:
            modified_reward = 25

        # Reduce illegal action penalty
        elif original_reward == -10:
            modified_reward = -5

        # Bonus for successful pickup
        if pass_loc < 4 and next_pass_loc == 4:
            modified_reward += 2

        # Distance-based shaping
        shaping_reward = 0.0

        if pass_loc < 4:  # Moving toward passenger
            pass_row, pass_col = self._get_location_coords(pass_loc)
            old_dist = self._manhattan_distance(taxi_row, taxi_col, pass_row, pass_col)
            new_dist = self._manhattan_distance(next_taxi_row, next_taxi_col, pass_row, pass_col)

            if new_dist < old_dist:
                shaping_reward += 0.2
            elif new_dist > old_dist:
                shaping_reward -= 0.2

        elif pass_loc == 4:  # Moving toward destination
            dest_row, dest_col = self._get_location_coords(dest)
            old_dist = self._manhattan_distance(taxi_row, taxi_col, dest_row, dest_col)
            new_dist = self._manhattan_distance(next_taxi_row, next_taxi_col, dest_row, dest_col)

            if new_dist < old_dist:
                shaping_reward += 0.2
            elif new_dist > old_dist:
                shaping_reward -= 0.2

        return modified_reward + shaping_reward


# Dictionary of available reward functions for easy access
REWARD_FUNCTIONS = {
    "default": DefaultReward,
    "distance_based": DistanceBasedReward,
    "modified_penalty": ModifiedPenaltyReward,
    "enhanced": EnhancedReward,
}


def get_reward_wrapper(name: str):
    """Get reward wrapper class by name.

    Args:
        name: Name of reward function

    Returns:
        Reward wrapper class

    Raises:
        ValueError: If reward function name is not found
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]
