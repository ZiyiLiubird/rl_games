import os
import os.path as osp
import sys

parent_path = osp.dirname(__file__)
# print(parent_path)
sys.path.append(parent_path)

import math
from copy import deepcopy
from collections import OrderedDict, deque
import time
import random
import numpy as np
from gym.spaces import Discrete, Box, Tuple
from rl_games.envs.zhikong import comm_interface
from .util import init_info, obs_feature_list, continuous_act_feature_list

import math
from copy import deepcopy
from collections import OrderedDict, deque
import time
import random
import numpy as np
from gym.spaces import Discrete, Box, Tuple
from rl_games.envs.zhikong import comm_interface
from .util import init_info, obs_feature_list, continuous_act_feature_list

class BaseEnv(object):
    def __init__(self, **kwargs):
        """
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.

        """
        self.ip = kwargs.get('ip', '127.0.1.1')
        self.port = kwargs.get('worker_index', 888) + 8000
        self.excute_path =kwargs['excute_path']
        self.playmode = kwargs.get('playmode', 0)
        self.n_agents = kwargs.get('num_agents', 2)
        self.scenes = kwargs.get('scenes', 1)
        self.red_agents_num = kwargs.get('red_agents_num', 1)
        self.blue_agents_num = kwargs.get('blue_agents_num', 1)
        self.setting = kwargs.get("setting", 0)
        self.num_agents = self.red_agents_num
        self.apply_agent_ids = True
        self.concat_infos = True
        print(f"Red Agents Num: {self.red_agents_num}")
        print(f"Blue Agents Num: {self.blue_agents_num}")
        assert (self.n_agents % 2 == 0), ("We only support N vs N now.")
        self.excute_cmd = f'{self.excute_path} ip={self.ip} port={self.port} ' \
                          f'PlayMode={self.playmode} RedNum={self.red_agents_num} ' \
                          f'BlueNum={self.blue_agents_num} Scenes={self.scenes}'
        print(f"Starting environments...")
        print(f"{self.excute_cmd}")
        self.unity = os.popen(self.excute_cmd)
        time.sleep(12)

        self.env = comm_interface.env(self.ip, self.port, self.playmode)
        self._episode_steps = 0
        self.episode_limit =kwargs.get("episode_max_length", 1000)

        # setup agents
        self.red_agents = ["red_" + str(i) for i in range(int(self.red_agents_num))]
        self.blue_agents = ["blue_" + str(i) for i in range(int(self.blue_agents_num))]
        self.camp_2_agents = {}
        self.camp_2_agents['red'] = self.red_agents
        self.camp_2_agents['blue'] = self.blue_agents
        self.camp_2_agents_num = {}
        self.camp_2_agents_num['red'] = self.red_agents_num
        self.camp_2_agents_num['blue'] = self.blue_agents_num
        self.possible_agents = self.red_agents + self.blue_agents
        self.agents = self.possible_agents
        print(f"possible agents: {self.possible_agents}")
        self.red_ni_mapping = OrderedDict(
                    zip(self.red_agents, list(range(self.red_agents_num))))
        self.blue_ni_mapping = OrderedDict(
                    zip(self.blue_agents, list(range(self.blue_agents_num))))

        # death tracking
        self.red_death = np.zeros((self.red_agents_num), dtype=int)
        self.red_death_missile = np.zeros((self.red_agents_num), dtype=int)
        self.blue_death = np.zeros((self.blue_agents_num), dtype=int)
        self.blue_death_missile = np.zeros((self.blue_agents_num), dtype=int)

        # setup dict_init
        self.dict_init = init_info(self.n_agents // 2, reset=False, seed=self.setting)
        self.dict_reset = init_info(self.n_agents//2, seed=self.setting)

        self.act_feature_list = continuous_act_feature_list
        self.action_space = Box(low=np.array([-1.0, -1.0, 0.5], dtype=np.float32),
                                high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
                                shape=(3,))

        # setup actions dict
        self.current_actions = OrderedDict(red=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.red_agents]),
                                    blue=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.blue_agents]))

        for agent_name in self.red_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['red'][agent_name][act_feature] = 0.

        for agent_name in self.blue_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['blue'][agent_name][act_feature] = 0.


    def reset(self):
        raise NotImplementedError
    
    def weapon_actions(self, ego_actions, camp='red'):
        """
        switch missile and weapon launch
        """
        camp_another = 'blue' if camp == 'red' else 'red'

        agents = self.camp_2_agents[camp]
        # srmissile_nums = np.zeros((self.red_agents_num, 1), dtype=np.float32)
        for al_id, agent_name in enumerate(agents):
            if int(self.prev_obs_dict[camp][agent_name]['DeathEvent']) != 99:
                continue
            srmissile_nums = self.prev_obs_dict[camp][agent_name]['SRAAMCurrentNum']
            armissile_nums = self.prev_obs_dict[camp][agent_name]['AMRAAMCurrentNum']
            aim_mode = self.prev_obs_dict[camp][agent_name]['AimMode']

            dis_min = np.min(
                [np.linalg.norm(np.array([self.GPS_to_xyz(self.obs_dict[camp][agent_name]['position/lat-geod-deg'],
                                                          self.obs_dict[camp][agent_name]['position/long-gc-deg'],
                                                          self.obs_dict[camp][agent_name]['position/h-sl-ft'])]) -
                                np.array([self.GPS_to_xyz(
                                    self.obs_dict[camp_another][agent_another]['position/lat-geod-deg'],
                                    self.obs_dict[camp_another][agent_another]['position/long-gc-deg'],
                                    self.obs_dict[camp_another][agent_another]['position/h-sl-ft'])]))
                 for agent_another in self.camp_2_agents[camp_another]])

            # ego_actions[al_id][-1] : 切换->1    不切换->0                   aim_mode: 0--srmissile   1--armissile
            if dis_min >= 10000. and armissile_nums != 0:
                ego_actions[al_id][0] = 1 if aim_mode == 0 else 0
            elif dis_min >= 10000. and armissile_nums == 0:
                ego_actions[al_id][0] = 1 if aim_mode == 1 else 0
            elif dis_min < 10000. and srmissile_nums != 0:
                ego_actions[al_id][0] = 1 if aim_mode == 1 else 0
            elif dis_min < 10000. and srmissile_nums == 0:
                ego_actions[al_id][0] = 1 if aim_mode == 0 else 0
            else:
                ego_actions[al_id][0] = 0

            # ego_actions[al_id][-2]  :  不发射-> 0    发射导弹->1    发射子弹->2
            # 如果视野内存在敌机且距离小于2km，则开启机枪
            if '99' not in str(self.prev_obs_dict[camp][agent_name]['TargetIntoView']) and \
                dis_min < 2000 and int(srmissile_nums) == 0:
                ego_actions[al_id][1] = 2
            # 如果远程导弹锁定敌方时，且距离在10-40km内，则发射
            elif '99' not in str(self.prev_obs_dict[camp][agent_name]['AMRAAMlockedTarget']) and \
                10000. <= dis_min < 40000.:
                ego_actions[al_id][1] = 1
            # 如果近程导弹锁定敌方时，且距离小于10km，则发射
            elif '99' not in str(self.prev_obs_dict[camp][agent_name]['SRAAMTargetLocked']) and \
                dis_min < 10000.:
                ego_actions[al_id][1] = 1
        return ego_actions

    def step(self, action):
        raise NotImplementedError

    def process_actions(self, ego_action, op_action,
                        ego_weapon_action, op_weapon_action):

        for i in range(int(self.red_agents_num)):
            self.current_actions['red']['red_'+str(i)]["fcs/aileron-cmd-norm"] = float(ego_action[i][0])
            self.current_actions['red']['red_'+str(i)]["fcs/elevator-cmd-norm"] = float(ego_action[i][1])
            self.current_actions['red']['red_'+str(i)]["fcs/throttle-cmd-norm"] = float(ego_action[i][2])
            self.current_actions['red']['red_'+str(i)]["switch-missile"] =  float(ego_weapon_action[i][0])
            self.current_actions['red']['red_'+str(i)]["fcs/weapon-launch"] =  float(ego_weapon_action[i][1])

        for i in range(int(self.blue_agents_num)):
            self.current_actions['blue']['blue_'+str(i)]["fcs/aileron-cmd-norm"] = float(op_action[i][0])
            self.current_actions['blue']['blue_'+str(i)]["fcs/elevator-cmd-norm"] = float(op_action[i][1])
            self.current_actions['blue']['blue_'+str(i)]["fcs/throttle-cmd-norm"] = float(op_action[i][2])
            self.current_actions['blue']['blue_'+str(i)]["switch-missile"] = float(op_weapon_action[i][0])
            self.current_actions['blue']['blue_'+str(i)]["fcs/weapon-launch"] = float(op_weapon_action[i][1])

    def get_obs(self):
        raise NotImplementedError

    def get_obs_op(self, camp='blue'):
        """
        opponent info including health and vel infos.
        """
        raise NotImplementedError

    def get_obs_alley(self, ego_agent_name,
                      ego_x, ego_y, ego_z, camp='red'):
        raise NotImplementedError

    def get_obs_agent(self, agent_name, camp='red'):
        raise NotImplementedError
