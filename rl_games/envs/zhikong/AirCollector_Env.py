from distutils.log import info
import enum
import os.path as osp
import sys

parent_path = osp.dirname(__file__)
# print(parent_path)
sys.path.append(parent_path)

import math
from copy import deepcopy
from collections import OrderedDict, deque, namedtuple
import time
import random
from ale_py import os
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
# from rl_games.envs.zhikong import comm_interface
from zhikong import comm_interface
from .util import init_info, obs_feature_list, act_feature_list

action_aileron = np.linspace(-1., 1., 9) # fcs/aileron-cmd-norm
action_elevator = np.linspace(-1., 1., 9) # fcs/elevator-cmd-norm
action_rudder = np.linspace(-1., 1., 9) # fcs/rudder-cmd-norm
action_throttle = np.linspace(0., 1., 5)  # fcs/throttle-cmd-norm



class AirCollectEnv(object):

    def __init__(self, **kwargs):
        # setup env
        self.ip = kwargs.get('ip', '127.0.1.1')
        self.port = kwargs.get('worker_index', 888) + 8000
        self.excute_path =kwargs['excute_path']
        self.playmode = kwargs.get('playmode', 0)
        self.n_agents = kwargs.get('num_agents', 2)
        self.scenes = kwargs.get('scenes', 1)
        self.red_agents_num = kwargs.get('red_agents_num', 1)
        self.blue_agents_num = kwargs.get('blue_agents_num', 1)
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
        self.action_space_type = kwargs.get('action_space_type', "MultiDiscrete")
        self._episode_steps = 0
        self.episode_limit =kwargs.get("episode_max_length", 500)
        self.change_target = kwargs.get("change_target", False)
        self.single_agent_mode = kwargs.get("single_agent_mode", False)
        self.win_record = deque(maxlen=30)

        self.min_height = kwargs.get("min_height", 1000)
        # reward
        # death tracking
        self.red_death = np.zeros((self.red_agents_num), dtype=int)
        self.red_death_missile = np.zeros((self.red_agents_num), dtype=int)
        self.blue_death = np.zeros((self.blue_agents_num), dtype=int)
        self.blue_death_missile = np.zeros((self.blue_agents_num), dtype=int)

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

        # setup dict_init
        self.dict_init = init_info(self.n_agents // 2, reset=False)
        self.dict_reset = init_info(self.n_agents//2)

        # define action and observation spaces
        assert (self.action_space_type == "MultiDiscrete"), "We only support MultiDiscrete action space now"

        # (aileron)：9
        # (elevator): 9
        # (rudder): 9
        # (throttle): 5
        # (weapon-launch): 2
        # (switch-missile)： 2
        # (change-target): 0/1/12/012/0134. 99 default.
        if self.change_target:
            self.act_feature_list = act_feature_list
            self.act_feature_list.append("change-target")
            self.action_space = Tuple((Discrete(9), Discrete(9),
                                    Discrete(9), Discrete(5), Discrete(2), Discrete(2),Discrete(5)))
        else:
            self.act_feature_list = act_feature_list
            self.action_space = Tuple((Discrete(9), Discrete(9),
                                    Discrete(9), Discrete(5), Discrete(2), Discrete(2)))

        self.action_map = {}
        self.action_map["fcs/aileron-cmd-norm"] = action_aileron
        self.action_map["fcs/elevator-cmd-norm"] = action_elevator
        self.action_map["fcs/rudder-cmd-norm"] = action_rudder
        self.action_map["fcs/throttle-cmd-norm"] = action_throttle

        # 42 obs
        shape = 91
        if self.red_agents_num == 1:
            shape = 91
        elif self.red_agents_num == 2:
            shape = 173
        elif self.red_agents_num == 3:
            shape = 255
        self.observation_spaces = [
            Box(low=np.float32(-np.inf), high=np.float32(np.inf),
                shape=(shape, ), dtype=np.float32) for _ in range(self.n_agents)]
        self.observation_space = self.observation_spaces[0]

        self.obs_feature_list = obs_feature_list

        # setup actions dict
        self.current_actions = OrderedDict(red=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.red_agents]),
                                    blue=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.blue_agents]))

        for agent_name in self.red_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['red'][agent_name][act_feature] = 0.

        for agent_name in self.blue_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['blue'][agent_name][act_feature] = 0.

    def reset(self, init=False):
        self.red_death = np.zeros((self.red_agents_num), dtype=int)
        self.red_death_missile = np.zeros((self.red_agents_num), dtype=int)
        self.blue_death = np.zeros((self.blue_agents_num), dtype=int)
        self.blue_death_missile = np.zeros((self.blue_agents_num), dtype=int)

        self._episode_steps = 0
        if init:
            obs_dict = self.env.reset(self.dict_init)
        else:
            obs_dict = self.env.reset(self.dict_reset)
        print(f"env reset success !")

        return obs_dict

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""

        ego_action, op_action = actions[0], actions[1]
        ego_action = ego_action.reshape(self.red_agents_num, -1)
        op_action = op_action.reshape(self.blue_agents_num, -1)
        infos = {}
        dones = np.zeros(self.num_agents, dtype=bool)
        rewards = 0
        self.process_actions(ego_action, op_action)

        self.obs_dict = self.env.step(self.current_actions)

        actions_human = self.save_obs_and_actions()

        self._episode_steps += 1

        game_done = self.judge_done()
        if game_done:
            for i in range(self.num_agents):
                dones[i] = True
        elif self._episode_steps >= self.episode_limit:
            for i in range(self.num_agents):
                dones[i] = True
            

        return self.obs_dict, actions_human, dones

    def process_actions(self, ego_action, op_action):
        """
        ego_action: (num_agent, action_num)
        op_action: (num_op_agent, action_num)

        """
        for i in range(int(self.red_agents_num)):
            for j, feature in enumerate(self.act_feature_list):
                if feature in self.action_map:
                    self.current_actions['red']['red_'+str(i)][feature] = self.action_map[feature][int(ego_action[i][j])]
                else:
                    self.current_actions['red']['red_'+str(i)][feature] =float(ego_action[i][j])
        for i in range(int(self.blue_agents_num)):
            for j, feature in enumerate(self.act_feature_list):
                if feature in self.action_map:
                    self.current_actions['blue']['blue_'+str(i)][feature] = self.action_map[feature][int(op_action[i][j])]
                else:
                    self.current_actions['blue']['blue_'+str(i)][feature] = float(op_action[i][j])

    def save_obs_and_actions(self):
        actions = np.zeros((self.red_agents_num, 6), dtype=np.float32)

        for id, agent_name in enumerate(self.red_agents):
            actions[id, 0] = self.obs_dict['red'][agent_name]['fcs/aileron-cmd-norm']
            actions[id, 1] = self.obs_dict['red'][agent_name]['fcs/elevator-cmd-norm']
            actions[id, 2] = self.obs_dict['red'][agent_name]['fcs/rudder-cmd-norm']
            actions[id, 3] = self.obs_dict['red'][agent_name]['fcs/throttle-cmd-norm']

        return actions

    def judge_done(self):

        self.red_all_dead = False
        self.blue_all_dead = False
        self.red_all_dead_missile = False
        self.blue_all_dead_missile = False

        for i, agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][agent_name]['DeathEvent']) != 99:
                self.red_death[i] = 1
                if int(self.obs_dict['red'][agent_name]['DeathEvent']) != 0:
                    self.red_death_missile[i] = 1

        for i, agent_name in enumerate(self.blue_agents):
            if int(self.obs_dict['blue'][agent_name]['DeathEvent']) != 99:
                self.blue_death[i] = 1
                if int(self.obs_dict['blue'][agent_name]['DeathEvent']) != 0:
                    self.blue_death_missile[i] = 1

        if np.sum(self.red_death) == self.red_agents_num:
            self.red_all_dead = True
            # print(f"Red all Dead !!!")
        if np.sum(self.blue_death) == self.blue_agents_num:
            self.blue_all_dead = True
            # print(f"Blue all Dead !!!")
        if np.sum(self.red_death_missile) == self.red_agents_num:
            self.red_all_dead_missile = True
            # print(f"Red all Dead !!!")
        if np.sum(self.blue_death_missile) == self.blue_agents_num:
            self.blue_all_dead_missile = True

        return (self.red_all_dead and self.blue_all_dead)
