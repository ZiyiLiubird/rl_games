import torch
import numpy as np
from gym import spaces
from rl_games.envs.zhikong import AirCombat_ConEnv
from rl_games.utils.motion_lib import MotionLib


class AirCombatAMPEnv(AirCombat_ConEnv.AirCombatConEnv):
    def __init__(self, **kwargs):




        motion_file = kwargs['motion_file']
        self._load_motion(motion_file)

        super().__init__(**kwargs)



    def _load_motion(self, motion_file, imitate_type):
        self._motion_lib = MotionLib(motion_file, imitate_type)
        return
