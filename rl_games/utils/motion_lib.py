import os
import torch
import numpy as np


class MotionLib():
    def __init__(self, motion_file, imitate_type="s_s"):
        assert imitate_type in ["s_a", "s_s"]
        self._load_motions(motion_file)
        self.motion_ids = torch.arange(len(self.obses_list), dtype=torch.long)
        self.process_motions()
        self.obs_num = 21
        self.process_motions()

    def _load_motions(self, motion_file):
        self.obses_list = []
        self.actions_list = []
        if os.path.isdir(motion_file):
            for file_path in os.listdir(motion_file):
                trans = torch.load(file_path)
                self.obses_list.extend(trans['obs'])
                self.actions_list.extend(trans['actions'])
        else:
            trans = torch.load(motion_file)
            self.obses_list.extend(trans['obs'])
            self.actions_list.extend(trans['actions'])

    def process_motions(self):

        length = self.get_total_length()
        self.obses = np.zeros((length, self.obs_num), dtype=np.float32)
        self.actions = np.zeros((length, 3), dtype=np.float32)

        for i in range(length):
            self.obses[i, 0] = self.obses_list[i]['red']['red_0']['position/h-sl-ft']
            self.obses[i, 1] = self.obses_list[i]['red']['red_0']['attitude/pitch-rad']
            self.obses[i, 2] = self.obses_list[i]['red']['red_0']['attitude/roll-rad']
            self.obses[i, 3] = self.obses_list[i]['red']['red_0']['attitude/psi-deg']
            self.obses[i, 4] = self.obses_list[i]['red']['red_0']['aero/beta-deg']
            self.obses[i, 5] = self.obses_list[i]['red']['red_0']['velocities/u-fps']
            self.obses[i, 6] = self.obses_list[i]['red']['red_0']['velocities/v-fps']
            self.obses[i, 7] = self.obses_list[i]['red']['red_0']['velocities/w-fps']
            
            self.obses[i, 8] = self.obses_list[i]['red']['red_0']['velocities/v-north-fps']
            self.obses[i, 9] = self.obses_list[i]['red']['red_0']['velocities/v-east-fps']
            
            self.obses[i, 10] = self.obses_list[i]['red']['red_0']['velocities/v-down-fps']
            self.obses[i, 11] = self.obses_list[i]['red']['red_0']['velocities/p-rad_sec']
            self.obses[i, 12] = self.obses_list[i]['red']['red_0']['velocities/q-rad_sec']
            self.obses[i, 13] = self.obses_list[i]['red']['red_0']['velocities/r-rad_sec']
            self.obses[i, 14] = self.obses_list[i]['red']['red_0']['velocities/ve-fps']
            self.obses[i, 15] = self.obses_list[i]['red']['red_0']['velocities/h-dot-fps']
            
            self.obses[i, 16] = self.obses_list[i]['red']['red_0']['fcs/left-aileron-pos-norm']
            self.obses[i, 17] = self.obses_list[i]['red']['red_0']['fcs/right-aileron-pos-norm']
            self.obses[i, 18] = self.obses_list[i]['red']['red_0']['fcs/elevator-pos-norm']
            self.obses[i, 19] = self.obses_list[i]['red']['red_0']['fcs/rudder-pos-norm']
            self.obses[i, 20] = self.obses_list[i]['red']['red_0']['fcs/throttle-pos-norm']

            self.actions[i, 0] = self.actions_list[i][0]
            self.actions[i, 1] = self.actions_list[i][1]
            self.actions[i, 2] = self.actions_list[i][2]

        self.obs_act = np.concatenate([self.obses, self.actions], dim=-1)

    def get_total_length(self):
        return len(self.obses_list)
    
    def sample(self, num_samples):
        pass
        
    
            
