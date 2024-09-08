import uuid
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

event_flags_start = 0xD747
event_flags_end = 0xD7F6 # 0xD761 # 0xD886 temporarily lower event flag range for obs input
museum_ticket = (0xD754, 0)

class PokeRedEnv(Env):
    def __init__(
            self, gb_path, init_state,
            max_steps=2048*8, headless=True,
            action_frequency=24, downscale_factor=2):
        
        self.headless = headless
        self.init_state = init_state
        self.act_freq = action_frequency
        self.max_steps = max_steps
        self.downscale_factor = downscale_factor

        self.pyboy = PyBoy(
            gb_path,
            debugging=False,
            disable_input=False,
            window_type="headless" if self.headless else "SDL2",
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.reset_count = 0

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 5000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=self._get_obs()["screen"].shape, dtype=np.uint8)
            }
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=0, options={}):
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.step_count = 0
        self.reset_count += 1

        return self._get_obs(), {}

    def render(self, reduce_res=True):
        game_pixels_render = self.screen.screen_ndarray()[:,:,0]  # (144, 160)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(
                    game_pixels_render, (self.downscale_factor,self.downscale_factor))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        screen = self.render()
        observation = {
            "screen": screen,
        }
        return observation

    def step(self, action):
        self.run_action_on_emulator(action)
        self.step_count += 1
        obs = self._get_obs()
        done = self.step_count >= self.max_steps - 1
        return obs, 0, False, done, {}
    
    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        if self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action])
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)
