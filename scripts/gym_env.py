import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

XML_PATH="/home/shourya/rl2/urdf/model.xml"

class RotaryInvertedPendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", None],
        "render_fps": 50,
    }
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

    def _get_obs(self): pass
    def _get_reward(self): pass
    def reset(self): pass
    def step(self, action): pass
    def render(self): pass
    def close(self): pass