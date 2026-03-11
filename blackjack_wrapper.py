# -*- coding: ascii -*-

import gymnasium as gym
import numpy as np
from typing import Any

ObsType = tuple[int, int, bool]
InfoType = dict[str, Any]


class BlackjackWrapper:

    def __init__(self, env_class: Any = None, **blackjack_kwargs: Any) -> None:
        # instantiate the backing gym environment
        if env_class is not None:
            self.env = env_class(**blackjack_kwargs)
        else:
            self.env = gym.make("Blackjack-v1", **blackjack_kwargs)
        self.render_mode = blackjack_kwargs.get("render_mode")

    def close(self) -> None:
        self.env.close()

    def play_random_episode(self, verbose: bool = False) -> float:
        # play one episode using random actions
        obs, _ = self.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            if verbose:
                print(f"State={obs}")
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, _ = self.step(action)
            total_reward += reward
            if verbose and terminated:
                print(f"Episode finished with reward {total_reward}")
        return total_reward

    def render(self) -> np.ndarray | None:
        return self.env.render()

    def reset(
        self, *, seed: int | None = None, **kwargs: Any
    ) -> tuple[ObsType, InfoType]:
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, InfoType]:
        return self.env.step(action)
