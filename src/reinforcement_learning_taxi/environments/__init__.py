"""Environment wrappers and modifications."""

from reinforcement_learning_taxi.environments.observation_wrappers import (
    OneHotObservationWrapper,
    TaxiActionMaskWrapper,
    TaxiFeatureWrapper,
    make_taxi_env,
)
from reinforcement_learning_taxi.environments.reward_wrappers import (
    REWARD_FUNCTIONS,
    DefaultReward,
    DistanceBasedReward,
    EnhancedReward,
    ModifiedPenaltyReward,
    TaxiRewardWrapper,
    get_reward_wrapper,
)

__all__ = [
    # Observation wrappers
    "TaxiFeatureWrapper",
    "OneHotObservationWrapper",
    "TaxiActionMaskWrapper",
    "make_taxi_env",
    # Reward wrappers
    "TaxiRewardWrapper",
    "DefaultReward",
    "DistanceBasedReward",
    "ModifiedPenaltyReward",
    "EnhancedReward",
    "REWARD_FUNCTIONS",
    "get_reward_wrapper",
]
