# -*- coding: ascii -*-

import csv
import numpy as np
import os
from PIL import Image
from typing import Any


class BlackjackInterface:

    def __init__(self, config: dict[str, Any]) -> None:
        # lazy import to avoid circular dependency
        from blackjack_wrapper import BlackjackWrapper

        self.config = config
        self.task_completed = False
        self.last_outcome: dict[str, Any] | None = None

        # episode tracking
        self.episode = 1
        self.num_episodes = config.get("num_episodes", 100)
        self.reward_list = list()
        self.current_episode_hits = 0
        self.current_episode_sticks = 0

        # csv logging
        self.csv_filename = config.get("episode_csv_file", "blackjack_episodes.csv")
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode_number", "hits", "sticks", "episode_score"])

        # determine variant
        self.use_blackjack_42 = config.get("use_blackjack_42", False)
        self.bust_threshold = 42 if self.use_blackjack_42 else 21

        # create the blackjack environment
        if self.use_blackjack_42:
            from blackjack_modified_goal import BlackjackEnv

            print("Using Blackjack-42 variant with a goal of 42 instead of 21")
            self.env = BlackjackWrapper(
                env_class=BlackjackEnv,
                render_mode="rgb_array",
                natural=True,
            )
        else:
            self.env = BlackjackWrapper(
                render_mode="rgb_array",
                natural=True,
            )

        # initial reset
        obs, _ = self.env.reset(seed=42)
        print(
            f"Initial blackjack state: Player sum={obs[0]}, "
            f"Dealer showing={obs[1]}, Usable ace={obs[2]}"
        )
        self._needs_reset = False

    def _reset_episode(self) -> None:
        # reset episode counters and reseed the environment
        self.episode += 1
        self.current_episode_hits = 0
        self.current_episode_sticks = 0
        obs, _ = self.env.reset(seed=self.episode)
        print(
            f"\n=== Blackjack Episode {self.episode} ===\n"
            f"Starting state: Player sum={obs[0]}, "
            f"Dealer showing={obs[1]}, Usable ace={obs[2]}"
        )
        self._needs_reset = False

    def close(self) -> None:
        # compute and append per-column averages to the csv
        if os.path.exists(self.csv_filename):
            import pandas as pd

            df = pd.read_csv(self.csv_filename)
            if not df.empty:
                avg_hits = df["hits"].mean()
                avg_sticks = df["sticks"].mean()
                avg_scores = df["episode_score"].mean()
                with open(self.csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Average", avg_hits, avg_sticks, avg_scores])

        self.env.close()

    def execute_command(self, command: str) -> str:
        # auto-reset if previous episode ended
        if self._needs_reset:
            self._reset_episode()

        # parse action
        command_lower = command.lower()
        if "hit" in command_lower:
            print("Taking action: hit")
            action = 1
            self.current_episode_hits += 1
        elif "stick" in command_lower:
            print("Taking action: stick")
            action = 0
            self.current_episode_sticks += 1
        else:
            print(f"Invalid action in response: {command}")
            action = 2

        # step the environment and format the new state
        obs, reward, terminated, truncated, info = self.env.step(action)
        result = (
            f"New state: Player sum={obs[0]}, "
            f"Dealer showing={obs[1]}, Usable ace={obs[2]}"
        )
        print(result)

        if terminated or truncated:
            print(f"Episode finished with reward {reward}")
            self.reward_list.append(reward)
            avg = sum(self.reward_list) / len(self.reward_list)
            print(f"Average reward: {avg}")

            dealer_sum = info.get("dealer_sum", 0)
            dealer_bust = info.get("dealer_bust", False)
            player_bust = obs[0] > self.bust_threshold
            game_result = "win" if reward > 0 else "loss" if reward < 0 else "draw"

            self.last_outcome = {
                "type": "blackjack_outcome",
                "result": game_result,
                "reward": reward,
                "player_sum": obs[0],
                "dealer_sum": dealer_sum,
                "dealer_bust": dealer_bust,
                "player_bust": player_bust,
                "dealer_showing": obs[1],
                "usable_ace": bool(obs[2]),
                "game_number": len(self.reward_list),
                "keep_history": True,
            }

            # write csv row
            with open(self.csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.episode,
                        self.current_episode_hits,
                        self.current_episode_sticks,
                        reward,
                    ]
                )

            self._needs_reset = True

            # check if all episodes are done
            if self.episode >= self.num_episodes:
                self.task_completed = True

            result += (
                f" | Episode {self.episode} finished: {game_result} (reward={reward})"
            )

        return result

    def get_camera_images(self) -> dict[str, Image.Image]:
        # render the environment and convert to a PIL image
        render_result = self.env.render()
        if render_result is not None:
            image = Image.fromarray(render_result.astype(np.uint8))
        else:
            image = Image.new("RGB", (600, 400), (0, 100, 0))
        return {"blackjack_table": image}

    def get_pending_events(self) -> list:
        if self.last_outcome is not None:
            events = [self.last_outcome]
            self.last_outcome = None
            return events
        return list()
