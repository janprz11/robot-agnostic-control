# -*- coding: ascii -*-

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np
import os


def cmp(a: float, b: float) -> float:
    return float(a > b) - float(a < b)


# 1 = ace, 2-10 = number cards, jack/queen/king = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random: np.random.Generator) -> int:
    return int(np_random.choice(deck))


def draw_hand(np_random: np.random.Generator) -> list[int]:
    return [draw_card(np_random), draw_card(np_random)]


def is_bust(hand: list[int]) -> bool:
    return sum_hand(hand) > 42


def score(hand: list[int]) -> int:
    return 0 if is_bust(hand) else sum_hand(hand)


def sum_hand(hand: list[int]) -> int:
    # compute hand total, counting at most one ace as 11
    ace_count = hand.count(1)
    if ace_count == 0 or not usable_ace(hand):
        return sum(hand)
    # add 10 for one ace (only one ace can be counted as 11)
    return sum(hand) + 10


def usable_ace(hand: list[int]) -> int:
    # check whether any ace can be counted as 11 without busting
    ace_count = hand.count(1)
    if ace_count == 0:
        return 0
    hand_sum = sum(hand)
    for _ in range(ace_count):
        if hand_sum + 10 <= 42:
            return 1
        hand_sum -= 1
    return 0


class BlackjackEnv(gym.Env):
    # blackjack variant where the goal is to reach 42 instead of 21

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self, render_mode: str | None = None, natural: bool = False, sab: bool = False
    ) -> None:
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode

    def _get_dealer_obs(self) -> tuple[int, int, int]:
        return (sum_hand(self.dealer), self.dealer[0], usable_ace(self.dealer))

    def _get_obs(self) -> tuple[int, int, int]:
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def close(self) -> None:
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # import pygame; raises a helpful error if not installed
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        # compute layout dimensions and define colors
        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        # initialize pygame display or off-screen surface
        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path: str) -> pygame.Surface:
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path: str, size: int) -> pygame.font.Font:
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        # render dealer label, visible card, and hidden card
        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img: pygame.Surface) -> pygame.Surface:
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        # render player label and sum display
        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )

        # update display or return pixel array
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[tuple[int, int, int], dict]:
        # deal initial hands and assign a random suit to the dealer's top card
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        _, dealer_card_value, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, bool, dict]:
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand and score
            terminated = True
            player_score = score(self.player)
            while score(self.dealer) <= player_score and score(self.dealer) < 38:
                self.dealer.append(draw_card(self.np_random))
                if is_bust(self.dealer):
                    break
            reward = cmp(score(self.player), score(self.dealer))

        if self.render_mode == "human":
            self.render()

        dealer_info = {
            "dealer_sum": sum_hand(self.dealer),
            "dealer_bust": is_bust(self.dealer),
        }

        return self._get_obs(), reward, terminated, False, dealer_info


# pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
