import torch
import numpy as np
from rl_games.envs.zhikong import AirCombat_ConEnv
from rl_games.utils.motion_lib import MotionLib


class AirCombatAMPEnv(AirCombat_ConEnv.AirCombatConEnv):
    def __init__(self, **kwargs):
        self._num_amp_obs_steps = kwargs.get("numAMPObsSteps", 10)
        assert(self._num_amp_obs_steps >= 2)
        
        self.amp_batch_size = kwargs.get("amp_batch_size", 50)

        motion_file = kwargs['motion_file']
        self._load_motion(motion_file)

        super().__init__(**kwargs)

        self._amp_obs_buf = torch.zeros((self.amp_batch_size, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)




    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return

    def _load_motion(self, motion_file, imitate_type):
        self._motion_lib = MotionLib(motion_file, imitate_type)
        return
