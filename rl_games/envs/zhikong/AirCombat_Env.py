import os.path as osp
import sys

parent_path = osp.dirname(__file__)
print(parent_path)
sys.path.append(parent_path)

from copy import deepcopy
from collections import OrderedDict
import time
import random
from ale_py import os
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete
from zhikong import comm_interface
from .util import init_info, obs_feature_list, act_feature_list

action_aileron = np.linspace(-1., 1., 8) # fcs/aileron-cmd-norm
action_elevator = np.linspace(-1., 1., 8) # fcs/elevator-cmd-norm
action_rudder = np.linspace(-1., 1., 8) # fcs/rudder-cmd-norm
action_throttle = np.linspace(0., 1., 4)  # fcs/throttle-cmd-norm
# actions_map = dict()
# idx = 0
# for i in action0:
#     for j in action1:
#         for k in action2:
#             for z in action3:
#                 actions_map[idx] = [np.round(i, decimals=2), np.round(j, decimals=2), np.round(k, decimals=2), np.round(z, decimals=2)]
#                 idx += 1

class AirCombatEnv(object):

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
        self.episode_limit =kwargs.get("limit", 1000)

        # setup agents
        self.red_agents = ["red_" + str(i) for i in range(int(self.red_agents_num))]
        self.blue_agents = ["blue_" + str(i) for i in range(int(self.blue_agents_num))]
        self.possible_agents = self.red_agents + self.blue_agents
        self.agents = self.possible_agents
        print(f"possible agents: {self.possible_agents}")
        self.red_ni_mapping = OrderedDict(
                    zip(self.red_agents, list(range(self.red_agents_num))))
        self.blue_ni_mapping = OrderedDict(
                    zip(self.blue_agents, list(range(self.blue_agents_num))))

        # self.agent_ni_mapping = OrderedDict(
        #             zip(self.possible_agents, list(range(self.n_agents)))
        #         )
        # self.agent_in_mapping = OrderedDict(
        #             zip(list(range(self.n_agents)), self.possible_agents)
        #         )

        # setup dict_init
        self.dict_init = init_info(self.n_agents // 2, reset=False)
        self.dict_reset = init_info(self.n_agents//2)

        # define action and observation spaces
        assert (self.action_space_type == "MultiDiscrete"), "We only support MultiDiscrete action space now"
        self.action_spaces = [MultiDiscrete([8, 8, 8, 4, 3, 2]) for _ in range(self.n_agents)]
        self.action_space = self.action_spaces[0]

        # 42 obs
        self.observation_spaces = [
            Box(low=np.float32(-np.inf), high=np.float32(np.inf),
                shape=(42, ), dtype=np.float32) for _ in range(self.n_agents)]
        self.observation_space = self.observation_spaces[0]

        self.action_space_dict = {
            agent: space for agent, space in zip(self.agents, self.action_spaces)
        }
        self.observation_space_dict = {
            agent: space for agent, space in zip(self.agents, self.observation_spaces)
        }

        self.obs_feature_list = obs_feature_list
        self.act_feature_list = act_feature_list

        # setup actions dict
        self.current_actions = OrderedDict(red=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.red_agents]),
                                    blue=OrderedDict([(agent_name, OrderedDict()) for agent_name in self.blue_agents]))

        for agent_name in self.red_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['red'][agent_name][act_feature] = 0.

        for agent_name in self.blue_agents:
            for act_feature in self.act_feature_list:
                self.current_actions['blue'][agent_name][act_feature] = 0.

        # setup attack variables
        self.locked_time = 0
        self.old_oracle = None
        self.cur_oracle = None

    def reset(self, init=False):
        print(f"reset start...")
        self._episode_steps = 0
        if init:
            print(self.dict_init)
            obs_dict = self.env.reset(self.dict_init)
        else:
            print(self.dict_reset)
            obs_dict = self.env.reset(self.dict_reset)
        print(f"env reset success !")
        agent_obs = self.get_obs(obs_dict)
        self.dones = set()

        print(f"reset success!")
        return agent_obs

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""

        ego_action, op_action = actions[0], actions[1]

        terminated = False
        bad_transition = False
        infos = [{} for i in range(self.num_agents)]
        dones = np.zeros(self.num_agents, dtype=bool)

        rewards = 0

        self.process_actions(ego_action, op_action)

        obs_dict_next = self.env.step(self.current_actions)
        if self.judge_done(obs_dict_next):
            terminated = True
            for i in range(self.num_agents):
                dones[i] = True
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            bad_transition = True

        obs, obs_op = self.get_obs(obs_dict_next)
        global_state = self.get_state(obs_dict_next)

        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = self._preproc_obs(obs, obs_op)
        obs_dict['states'] = global_state

        # rewards = self.reward_battle(obs_dict_next)

        print(f"step success")

        return obs_dict, rewards, dones, infos

    def process_actions(self, ego_action, op_action):
        for i in range(int(self.red_agents_num)):
            for j, feature in enumerate(self.act_feature_list):
                self.current_actions['red']['red_'+str(i)][feature] = float(ego_action[i][j])
        for i in range(int(self.blue_agents_num)):
            for j, feature in enumerate(self.act_feature_list):
                self.current_actions['blue']['blue_'+str(i)][feature] = float(op_action[i][j])

    def _preproc_obs(self, obs, obs_op):
        # todo: remove from self
        if self.apply_agent_ids:
            num_agents = self.num_agents
            obs = np.array(obs)
            obs_op = np.array(obs_op)
            red_ids = np.eye(num_agents, dtype=np.float32)
            blue_ids = np.eye(self.blue_agents_num, dtype=np.float32)
            obs = np.concatenate([obs, red_ids], axis=-1)
            obs_op = np.concatenate([obs_op, blue_ids], axis=-1)

        return obs, obs_op

    def get_obs(self, agents_dict):

        ego_obs_dict = agents_dict['red']
        op_obs_dict = agents_dict['blue']

        obs = np.zeros((self.red_agents_num, self.observation_space.shape[0]), dtype=np.float32)
        obs_op = np.zeros((self.blue_agents_num, self.observation_space.shape[0]), dtype=np.float32)

        for agent_name in ego_obs_dict.keys():
            if ego_obs_dict[agent_name]['DeathEvent'] != 99.0:
                continue
            for i, feature in enumerate(self.obs_feature_list):
                obs[self.red_ni_mapping[agent_name]][i] = ego_obs_dict[agent_name][feature]

        for agent_name in op_obs_dict.keys():
            if op_obs_dict[agent_name]['DeathEvent'] != 99.0:
                continue
            for i, feature in enumerate(self.obs_feature_list):
                obs_op[self.blue_ni_mapping[agent_name]][i] = op_obs_dict[agent_name][feature]

        return obs, obs_op

    def get_state(self, agents_dict):
        return None

    def judge_done(self, agents_dict):
        red_all_dead = True
        blue_all_dead = True
        # print(f"************obs_dict inner*******************")
        # print(agents_dict)
        for key in agents_dict['red'].keys():
            print(f"red: {key} {agents_dict['red'][key].keys()}")
            if agents_dict['red'][key]['DeathEvent'] == 99.0:
                print(f"red death event: {agents_dict['red'][key]['DeathEvent']}")
                red_all_dead = False
                break
        if red_all_dead:
            print("Red all Dead!")
            return red_all_dead
        for key in agents_dict['blue'].keys():
            # print(f"blue: keys {agents_dict['blue'][key].keys()}")
            if agents_dict['blue'][key]['DeathEvent'] == 99.0:
                blue_all_dead = False
                break
        if blue_all_dead:
            print("Blue all Dead!")
        return blue_all_dead
    
    def seed(self, seed=1):
        random.seed(seed)
        np.random.seed(seed=seed)

    def action_space_sample(self, agent_ids: list = None):
        if agent_ids is None:
            agent_ids = list(range(len(self.agents)))
        actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}

        return actions

    def reward_battle(self, agents_dict):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can calculate reward according to the obs

        Returns:
            calculated reward
        """
        if self.reward_sparse:
            return 0
        
        
        
        raise NotImplementedError

    def reset_var(self):
        self.locked_time = 0
        self.old_oracle = None
        self.cur_oracle = None
