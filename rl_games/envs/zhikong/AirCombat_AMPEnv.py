import torch
import numpy as np
from rl_games.envs.zhikong import AirCombat_ConEnv
from rl_games.utils.motion_lib import MotionLib


class AirCombatAMPEnv(AirCombat_ConEnv.AirCombatConEnv):
    def __init__(self, **kwargs):
        self._num_amp_obs_steps = kwargs.get("numAMPObsSteps", 10)
        assert(self._num_amp_obs_steps >= 2)

        motion_file = kwargs['motion_file']
        self._load_motion(motion_file)

        super().__init__(**kwargs)

        self._amp_obs_buf = torch.zeros((self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return

    def _load_motion(self, motion_file, imitate_type):
        self._motion_lib = MotionLib(motion_file, imitate_type)
        return
