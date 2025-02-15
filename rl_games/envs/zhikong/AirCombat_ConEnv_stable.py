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
# from zhikong import comm_interface
from .util import init_info, obs_feature_list, continuous_act_feature_list
import time


class AirCombatConEnv(object):
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
        self.setting = kwargs.get("setting", 0)
        print(f"Red Agents Num: {self.red_agents_num}")
        print(f"Blue Agents Num: {self.blue_agents_num}")
        assert (self.n_agents % 2 == 0), ("We only support N vs N now.")
        self.excute_cmd = f'{self.excute_path} ip={self.ip} port={self.port} ' \
                          f'PlayMode={self.playmode} RedNum={self.red_agents_num} ' \
                          f'BlueNum={self.blue_agents_num} Scenes={self.scenes}'
        print(f"Starting environments...")
        print(f"{self.excute_cmd}")
        self.unity = os.popen(self.excute_cmd)
        time.sleep(15)

        self.env = comm_interface.env(self.ip, self.port, self.playmode)
        self._episode_steps = 0
        self.episode_limit =kwargs.get("episode_max_length", 1000)
        self.single_agent_mode = kwargs.get("single_agent_mode", False)
        self.win_record = deque(maxlen=30)

        self.min_height = kwargs.get("min_height", 10000)
        # reward
        self.cum_rewards = 0
        self.reward_win = kwargs.get("reward_win", 100)
        self.reward_sparse = kwargs.get("reward_sparse", False)
        self.reward_negative_scale = kwargs.get("reward_negative_scale", 0.5)
        self.reward_death_value = kwargs.get("reward_death_value", 10)
        self.reward_scale = kwargs.get("reward_scale", True)
        self.reward_scale_rate = kwargs.get("reward_scale_rate", 20)
        self.reward_defeat = kwargs.get("reward_defeat", -1)
        self.reward_only_positive = kwargs.get("reward_only_positive", False)
        self.max_reward = (self.reward_win + 2 * self.reward_death_value * self.blue_agents_num
                           + 200*self.blue_agents_num + 200 + 20)
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
        self.dict_init = init_info(self.n_agents // 2, reset=False, seed=self.setting)
        self.dict_reset = init_info(self.n_agents//2, seed=self.setting)

        # define action and observation spaces
        # (aileron)：9
        # (elevator): 9
        # (rudder): 9
        # (throttle): 5
        # (weapon-launch): 2
        # (switch-missile)： 2
        # (change-target): 0/1/12/012/0134. 99 default.

        self.act_feature_list = continuous_act_feature_list
        self.action_space = Box(low=np.array([-1.0, -1.0, 0.5], dtype=np.float32),
                                high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
                                shape=(3,))

        # 42 obs
        shape = 81
        if self.red_agents_num == 1:
            shape = 81
        elif self.red_agents_num == 2:
            shape = 173 - 5  # 可能有问题 
        elif self.red_agents_num == 3:
            shape = 255 - 5  # 可能有问题
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
        # print(f"reset start...")
        # print(f"init: {init}")
        self.red_death = np.zeros((self.red_agents_num), dtype=int)
        self.red_death_missile = np.zeros((self.red_agents_num), dtype=int)
        self.blue_death = np.zeros((self.blue_agents_num), dtype=int)
        self.blue_death_missile = np.zeros((self.blue_agents_num), dtype=int)
        self.cum_rewards = 0
        self._episode_steps = 0
        self.record = False
        if init:
            self.obs_dict = self.env.reset(self.dict_init)
            self.prev_obs_dict = None
        else:
            self.obs_dict = self.env.reset(self.dict_reset)
            self.prev_obs_dict = None
        # print(f"env reset success !")
        obs, obs_op = self.get_obs()
        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] =obs, obs_op

        return obs_dict

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
            if int(self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange']) == 99:
                continue
            # fly_ids = self.into_view(self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange'])
            print(f"TargetEnterAttackRange: {self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange']}")
            fly_ids = self._view_sin(self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange'], base=0)
            fly_ids = np.array(fly_ids)
            # goals = fly_ids.nonzero()[0]
            goals = fly_ids
            print(f"!!! goals !!! {goals}")
            if len(goals) == 0:
                continue
            ego_x, ego_y, ego_z = self.GPS_to_xyz(self.prev_obs_dict[camp][agent_name]['position/lat-geod-deg'],
                                                  self.prev_obs_dict[camp][agent_name]['position/long-gc-deg'],
                                                  self.prev_obs_dict[camp][agent_name]['position/h-sl-ft'])
            srmissile_nums = self.prev_obs_dict[camp][agent_name]['SRAAMCurrentNum']
            armissile_nums = self.prev_obs_dict[camp][agent_name]['AMRAAMCurrentNum']
            bullet_nums = self.prev_obs_dict[camp][agent_name]['BulletCurrentNum']
            aim_mode = self.prev_obs_dict[camp][agent_name]['AimMode']
            sr_locked = self.prev_obs_dict[camp][agent_name]['SRAAMTargetLocked']
            ar_locked = self.prev_obs_dict[camp][agent_name]['AMRAAMlockedTarget']
            if len(goals) == 1:
                item = goals.item()
                enemy = 'blue_'+str(item) if camp_another == 'blue' else 'red_'+str(item)
                enemy_x, enemy_y,enemy_z = self.GPS_to_xyz(
                    self.prev_obs_dict[camp_another][enemy]['position/lat-geod-deg'],
                    self.prev_obs_dict[camp_another][enemy]['position/long-gc-deg'],
                    self.prev_obs_dict[camp_another][enemy]['position/h-sl-ft'])
                dist = np.linalg.norm(np.array([enemy_x-ego_x, enemy_y-ego_y, enemy_z-ego_z]))
                print(f"!!! dist !!! {dist}")
                if dist < 2000:
                    if bullet_nums > 0:
                     ego_actions[al_id][1] = 2
                    else:
                        continue
                elif dist <= 12000:
                    if aim_mode == 0:
                        if srmissile_nums != 0 and sr_locked != 99:
                            ego_actions[al_id][1] = 1
                        else:
                            continue
                    else:
                        ego_actions[al_id][0] = 1
                else:
                    if aim_mode == 1:
                        if armissile_nums != 0 and ar_locked != 99:
                            ego_actions[al_id][1] = 1
                    else:
                        ego_actions[al_id][0] = 1

            else: # goals > 1
                if aim_mode == 0:
                    if armissile_nums > 0:
                        ego_actions[al_id][0] = 1
                    elif srmissile_nums > 0:
                        if sr_locked:
                            ego_actions[al_id][1] = 1
                elif aim_mode == 1:
                    if armissile_nums > 0:
                        if ar_locked:
                            ego_actions[al_id][1] = 1
                    elif srmissile_nums > 0:
                        ego_actions[al_id][0] = 1

        return ego_actions

    # def weapon_actions(self, ego_actions, camp='red'):
    #     """
    #     switch missile and weapon launch
    #     """
    #     agents = self.camp_2_agents[camp]
    #     # srmissile_nums = np.zeros((self.red_agents_num, 1), dtype=np.float32)
    #     for al_id, agent_name in enumerate(agents):
    #         if int(self.prev_obs_dict[camp][agent_name]['DeathEvent']) != 99:
    #             continue
    #         srmissile_nums = self.prev_obs_dict[camp][agent_name]['SRAAMCurrentNum']
    #         armissile_nums = self.prev_obs_dict[camp][agent_name]['AMRAAMCurrentNum']
    #         aim_mode = self.prev_obs_dict[camp][agent_name]['AimMode']
    #         if srmissile_nums == 0 and armissile_nums != 0 and aim_mode == 0:
    #             ego_actions[al_id][0] = 1
    #         elif (armissile_nums == 0 and srmissile_nums != 0) and aim_mode == 1:
    #             ego_actions[al_id][0] = 1
    #         elif (armissile_nums != 0 and srmissile_nums !=0) and aim_mode == 0:
    #             ego_actions[al_id][0] = 1
    #         else:
    #             ego_actions[al_id][0] = 0
    #         sr_locked = self.prev_obs_dict[camp][agent_name]['SRAAMTargetLocked']
    #         ar_locked = self.prev_obs_dict[camp][agent_name]['AMRAAMlockedTarget']
    #         if '99' not in str(sr_locked) or '99' not in str(ar_locked):
    #             ego_actions[al_id][1] = 1
    #         else:
    #             ego_actions[al_id][1] = 0

    #     return ego_actions

    def step(self, action):
        """A single environment step. Returns reward, terminated, info."""

        self.prev_obs_dict = deepcopy(self.obs_dict)

        ego_action, op_action = action[0], action[1]
        ego_action = ego_action.reshape(self.red_agents_num, -1)
        op_action = op_action.reshape(self.blue_agents_num, -1)

        ego_weapon_actions = np.zeros((self.red_agents_num, 2), dtype=int)
        op_weapon_actions = np.zeros((self.red_agents_num, 2), dtype=int)

        ego_weapon_actions = self.weapon_actions(ego_weapon_actions, camp='red')
        op_weapon_actions = self.weapon_actions(op_weapon_actions, camp='blue')

        infos = {}
        dones = np.zeros(self.num_agents, dtype=bool)

        self.process_actions(ego_action, op_action, ego_weapon_actions, op_weapon_actions)

        self.obs_dict = self.env.step(self.current_actions)

        self._episode_steps += 1

        if self.single_agent_mode:
            reward = self.reward_single_agent_mode()
        else:
            # print("=========这是一个检查点==========")
            reward = self.reward_battle()

        game_done = self.judge_done()
        if game_done:
            for i in range(self.num_agents):
                dones[i] = True
        elif self._episode_steps >= self.episode_limit:
            for i in range(self.num_agents):
                dones[i] = True

        obs, obs_op = self.get_obs()

        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = obs, obs_op

        infos['win'] = False
        infos['lose'] = False
        infos['draw'] = False

        if not self.record:
            if self.blue_all_dead and not self.red_all_dead:
                if self.reward_sparse and not self.single_agent_mode:
                    reward = 1
                else:
                    reward += self.reward_win
                self.record = True
                infos['win'] = True
                infos['lose'] = False
                infos['draw'] = False
            elif self.red_all_dead:
                if self.reward_sparse and not self.single_agent_mode:
                    reward = -1
                else:
                    reward += self.reward_defeat
                self.record = True
                infos['win'] = False
                infos['lose'] = True
                infos['draw'] = False
        elif self._episode_steps >= self.episode_limit and not self.record:
                infos['win'] = False
                infos['lose'] = False
                infos['draw'] = True
                self.record = True
        else:
            infos['win'] = False
            infos['lose'] = False
            infos['draw'] = False

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
        self.cum_rewards += reward

        rewards = [np.array(reward, dtype=np.float32)] * self.num_agents

        return obs_dict, rewards, dones, infos

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

        ego_obs_dict = self.obs_dict['red']
        op_obs_dict = self.obs_dict['blue']

        obs = np.zeros((self.red_agents_num, self.observation_space.shape[0]), dtype=np.float32)
        obs_op = np.zeros((self.blue_agents_num, self.observation_space.shape[0]), dtype=np.float32)

        self.red_view_ego_obs = self.get_obs_ego(camp='red')
        self.red_view_op_obs = self.get_obs_op(camp='blue')
        self.blue_view_ego_obs = self.get_obs_ego(camp='blue')
        self.blue_view_op_obs = self.get_obs_op(camp='red')

        for agent_name in ego_obs_dict.keys():
            if int(ego_obs_dict[agent_name]['DeathEvent']) != 99:
                continue
            obs_agent = self.get_obs_agent(agent_name, camp='red')
            obs[self.red_ni_mapping[agent_name]] = obs_agent

        for agent_name in op_obs_dict.keys():
            if int(op_obs_dict[agent_name]['DeathEvent']) != 99:
                continue
            obs_agent = self.get_obs_agent(agent_name, camp='blue')
            obs_op[self.blue_ni_mapping[agent_name]] = obs_agent

        return obs, obs_op

    def get_obs_op(self, camp='blue'):
        """
        opponent info including health and vel infos.
        """
        op_infos = np.zeros((self.blue_agents_num, 16), dtype=np.float32)
        agents = self.camp_2_agents[camp]

        for i, agent_name in enumerate(agents):
            if int(self.obs_dict[camp][agent_name]['DeathEvent']) != 99:
                continue
            op_infos[i, 0] = self.obs_dict[camp][agent_name]['LifeCurrent']
            op_infos[i, 1] = self.obs_dict[camp][agent_name]['position/long-gc-deg']
            op_infos[i, 2] = self.obs_dict[camp][agent_name]['position/lat-geod-deg']
            op_infos[i, 3] = self.obs_dict[camp][agent_name]['velocities/u-fps']
            op_infos[i, 4] = self.obs_dict[camp][agent_name]['velocities/v-fps']
            op_infos[i, 5] = self.obs_dict[camp][agent_name]['velocities/w-fps']
            op_infos[i, 6] = self.obs_dict[camp][agent_name]['velocities/p-rad_sec']
            op_infos[i, 7] = self.obs_dict[camp][agent_name]['velocities/q-rad_sec']
            op_infos[i, 8] = self.obs_dict[camp][agent_name]['velocities/r-rad_sec']
            op_infos[i, 9] = self.obs_dict[camp][agent_name]['velocities/h-dot-fps']
            op_infos[i, 10] = self.obs_dict[camp][agent_name]['velocities/ve-fps']
            op_infos[i, 11] = self.obs_dict[camp][agent_name]['velocities/mach']
            op_infos[i, 12] = self.obs_dict[camp][agent_name]['position/h-sl-ft']
            op_infos[i, 13] = self.obs_dict[camp][agent_name]['attitude/pitch-rad']
            op_infos[i, 14] = self.obs_dict[camp][agent_name]['attitude/roll-rad']
            op_infos[i, 15] = self.obs_dict[camp][agent_name]['attitude/psi-deg']


        obs_op = op_infos.flatten()
        return obs_op

    def get_obs_ego(self, camp='red'):

        ego_agents = self.camp_2_agents[camp]
        ego_agent_num = self.camp_2_agents_num[camp]
        ego_infos = np.zeros((ego_agent_num, 20+10+27), dtype=np.float32)

        for i, name in enumerate(ego_agents):
            if int(self.obs_dict[camp][name]['DeathEvent']) != 99:
                continue
            # 缺失经纬度坐标？
            ego_infos[i, 0] = self.obs_dict[camp][name]['LifeCurrent']
            ego_infos[i, 1] = self.obs_dict[camp][name]['position/long-gc-deg']
            ego_infos[i, 2] = self.obs_dict[camp][name]['position/lat-geod-deg']
            ego_infos[i, 3] = self.obs_dict[camp][name]['position/h-sl-ft']
            ego_infos[i, 4] = self.obs_dict[camp][name]['attitude/pitch-rad']
            ego_infos[i, 5] = self.obs_dict[camp][name]['attitude/roll-rad']
            ego_infos[i, 6] = self.obs_dict[camp][name]['attitude/psi-deg']
            ego_infos[i, 7] = self.obs_dict[camp][name]['velocities/u-fps']
            ego_infos[i, 8] = self.obs_dict[camp][name]['velocities/v-fps']
            ego_infos[i, 9] = self.obs_dict[camp][name]['velocities/w-fps']
            ego_infos[i, 10] = self.obs_dict[camp][name]['velocities/p-rad_sec']
            ego_infos[i, 11] = self.obs_dict[camp][name]['velocities/q-rad_sec']
            ego_infos[i, 12] = self.obs_dict[camp][name]['velocities/ve-fps']
            ego_infos[i, 13] = self.obs_dict[camp][name]['velocities/h-dot-fps']
            ego_infos[i, 14] = self.obs_dict[camp][name]['velocities/mach']
            ego_infos[i, 15] = self.obs_dict[camp][name]['forces/load-factor']
            ego_infos[i, 16] = self.obs_dict[camp][name]['IsOutOfValidBattleArea']
            ego_infos[i, 17] = self.obs_dict[camp][name]['OutOfValidBattleAreaCurrentDuration']
            ego_infos[i, 18] = self.obs_dict[camp][name]['SRAAMCurrentNum']
            ego_infos[i, 19] = self.obs_dict[camp][name]['AMRAAMCurrentNum']
            ego_infos[i, 20+int(self.obs_dict[camp][name]['AimMode'])] = 1
            ego_infos[i, 22+int(self.obs_dict[camp][name]['SRAAM1_CanReload'])] = 1
            ego_infos[i, 24+int(self.obs_dict[camp][name]['SRAAM2_CanReload'])] = 1
            ego_infos[i, 26+int(self.obs_dict[camp][name]['AMRAAMCanReload'])] = 1
            ego_infos[i, 28+int(self.obs_dict[camp][name]['IfPresenceHitting'])] = 1

            if '99' not in str(self.obs_dict[camp][name]['TargetIntoView']):
                indices = self._view_sin(self.obs_dict[camp][name]['TargetIntoView'], base=30)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['AllyIntoView']):
                indices = self._view_sin(self.obs_dict[camp][name]['AllyIntoView'], base=35)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['TargetEnterAttackRange']):
                indices = self._view_sin(self.obs_dict[camp][name]['TargetEnterAttackRange'], base=40)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['SRAAMTargetLocked']):
                indices = self._view_sin(self.obs_dict[camp][name]['SRAAMTargetLocked'], base=45)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['AMRAAMlockedTarget']):
                indices = self._view_sin(self.obs_dict[camp][name]['AMRAAMlockedTarget'], base=50)
                ego_infos[i, indices] = 1

            ego_infos[i, 55 + int(self.obs_dict[camp][name]['MissileAlert'])] = 1

        return ego_infos.flatten()

    def get_obs_agent(self, agent_name, camp='red'):
        ctrl_state_feats = np.zeros((4), dtype=np.float32)
        ctrl_state_feats[0] = self.obs_dict[camp][agent_name]['fcs/left-aileron-pos-norm']
        ctrl_state_feats[1] = self.obs_dict[camp][agent_name]['fcs/right-aileron-pos-norm']
        ctrl_state_feats[2] = self.obs_dict[camp][agent_name]['fcs/elevator-pos-norm']
        ctrl_state_feats[3] = self.obs_dict[camp][agent_name]['fcs/throttle-pos-norm']

        if camp == 'red':
            agent_id_feats = np.zeros(self.red_agents_num, dtype=np.float32)
            ego_infos = self.red_view_ego_obs
            op_infos = self.red_view_op_obs
            op_camp = 'blue'
        else:
            agent_id_feats = np.zeros(self.blue_agents_num, dtype=np.float32)
            ego_infos = self.blue_view_ego_obs
            op_infos = self.blue_view_op_obs
            op_camp = 'red'

        # threat info
        threat_info = self._oracal_guiding_feature(agent_name, camp, op_camp)

        if self.apply_agent_ids:
            if camp == 'red':
                agent_id_feats[self.red_ni_mapping[agent_name]] = 1
            else:
                agent_id_feats[self.blue_ni_mapping[agent_name]] = 1

        obs_all = np.concatenate([
            ego_infos.flatten(), ctrl_state_feats.flatten(), op_infos.flatten(),
            threat_info.flatten(), agent_id_feats.flatten()
        ])

        # if camp == 'red':
        #     self.print(f"                                         oppo: {op_infos[1: 3]}")
        #     self.print(f"ctrl:{ego_infos[1: 4]}")

        return obs_all

    def _oracal_guiding_feature(self, agent_name, ego_camp, op_camp):

        op_agents = self.camp_2_agents[op_camp]

        threat_infos = np.zeros((self.blue_agents_num, 3), dtype=np.float32)

        ego_x, ego_y, ego_z = self.latitude_2_xyz(
            self.obs_dict[ego_camp][agent_name]['position/long-gc-deg'],
            self.obs_dict[ego_camp][agent_name]['position/lat-geod-deg'],
            self.obs_dict[ego_camp][agent_name]['position/h-sl-ft'] * 0.3048)

        for i, op_name in enumerate(op_agents):

            if int(self.obs_dict[op_camp][op_name]['DeathEvent']) != 99:
                continue
            op_x, op_y, op_z = self.latitude_2_xyz(
            self.obs_dict[op_camp][op_name]['position/long-gc-deg'],
            self.obs_dict[op_camp][op_name]['position/lat-geod-deg'],
            self.obs_dict[op_camp][op_name]['position/h-sl-ft'] * 0.3048)
            dist = np.array(math.sqrt((ego_x - op_x) ** 2 + (ego_y - op_y) ** 2 + (ego_z - op_z) ** 2), dtype=np.float32)
            threat_infos[i, 0] = dist

            elevation, azimuth = self.look_vector((ego_x, ego_y, ego_z), (op_x, op_y, op_z))
            d_elevation_red = elevation - self.obs_dict[ego_camp][agent_name]['attitude/pitch-rad'] * 180 / 3.14
            d_azimuth_red = azimuth - self.obs_dict[ego_camp][agent_name]['attitude/psi-deg']
            threat_infos[i, 1] = d_elevation_red
            threat_infos[i, 2] = d_azimuth_red
            # print('d_elevation_red: ', d_elevation_red)
            # print('d_azimuth_red', d_azimuth_red)
            # 对手视角，该部分按道理应该根据速度计算，但是为了简化计算过程，将位姿直接近似成了速度的方向
            elevation, azimuth = self.look_vector((op_x, op_y, op_z), (ego_x, ego_y, ego_z))
            d_elevation_blue = elevation - self.obs_dict[op_camp][op_name]['attitude/pitch-rad'] * 180 / 3.14
            d_azimuth_blue = azimuth - self.obs_dict[op_camp][op_name]['attitude/psi-deg']

        threat_info = threat_infos.flatten()

        # post_process_obs = [distance, d_elevation_red, d_azimuth_red,
        #                     d_elevation_blue, d_azimuth_blue]
        # # print('oracal_guiding', post_process_obs)
        # delta_h = (obs_red['position/h-sl-ft'] - obs_blue['position/h-sl-ft']) / \
        #           max(obs_red['position/h-sl-ft'], obs_blue['position/h-sl-ft'])
        # self.old_oracle = self.cur_oracle if self.cur_oracle else post_process_obs
        # self.cur_oracle = post_process_obs
        # self.old_delta_height = self.cur_delta_height if self.cur_delta_height else delta_h
        # self.cur_delta_height = delta_h
        # print(self.cur_delta_height, self.old_delta_height, self.cur_delta_height-self.old_delta_height)
        return threat_info

    def _view_sin(self, attribute, base):

        attribute = str(int(attribute))
        indices = []
        for i in attribute:
            if i == '.':
                continue
            indices.append(int(i)+base)
        return indices

    def get_state(self, agents_dict):
        return None

    def judge_done(self):

        self.red_all_dead = False
        self.blue_all_dead = False
        self.red_all_dead_missile = False
        self.blue_all_dead_missile = False
        # self.red_all_dead_missile_future = False
        # self.blue_all_dead_missile_future = False
        # present_hitting_red = np.zeros((self.red_agents_num), dtype=np.float32)
        # present_hitting_blue = np.zeros((self.blue_agents_num), dtype=np.float32)

        # present_hitting_red_num = 0
        # present_hitting_blue_num = 0

        for i, agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][agent_name]['DeathEvent']) != 99:
                self.red_death[i] = 1
                if int(self.obs_dict['red'][agent_name]['DeathEvent']) != 0:
                    self.red_death_missile[i] = 1
            # if self.obs_dict['red'][agent_name]['IfPresenceHitting'] == 1:
            #     present_hitting_red[i] = 1

        for i, agent_name in enumerate(self.blue_agents):
            if int(self.obs_dict['blue'][agent_name]['DeathEvent']) != 99:
                self.blue_death[i] = 1
                if int(self.obs_dict['blue'][agent_name]['DeathEvent']) != 0:
                    self.blue_death_missile[i] = 1
            # if self.obs_dict['blue'][agent_name]['IfPresenceHitting'] == 1:
            #     present_hitting_blue[i] = 1

        # present_hitting_red_num = np.sum(present_hitting_red)
        # present_hitting_blue_num = np.sum(present_hitting_blue)

        if np.sum(self.red_death) == self.red_agents_num:
            self.red_all_dead = True
        # elif (self.red_agents_num - np.sum(self.red_death_missile)) == present_hitting_blue_num:
        #     self.red_all_dead_missile_future = True
        if np.sum(self.blue_death) == self.blue_agents_num:
            self.blue_all_dead = True
        # elif (self.blue_agents_num - np.sum(self.blue_death_missile)) == present_hitting_red_num:
        #     self.blue_all_dead_missile_future = True

        if np.sum(self.red_death_missile) == self.red_agents_num:
            self.red_all_dead_missile = True
        if np.sum(self.blue_death_missile) == self.blue_agents_num:
            self.blue_all_dead_missile = True

        return (self.red_all_dead and self.blue_all_dead)

    def seed(self, seed=1):
        random.seed(seed)
        np.random.seed(seed=seed)

    # def reward_battle(self):
    #     """Reward function when self.reward_spare==False.
    #     Returns accumulative hit/shield point damage dealt to the enemy
    #     + reward_death_value per enemy unit killed, and, in case
    #     self.reward_only_positive == False, - (damage dealt to ally units
    #     + reward_death_value per ally unit killed) * self.reward_negative_scale
    #     """

    #     if self.reward_sparse:
    #         return 0

    #     neg_scale = self.reward_negative_scale
    #     delta_ally = 0
    #     delta_enemy = 0
    #     delta_deaths = 0
    #     locked = 0
    #     area = 0
    #     height = 0

    #     locked_adv, be_locked_adv = self.locked_reward()
    #     locked += np.sum(locked_adv) * 2
    #     locked -= np.sum(be_locked_adv) * neg_scale

    #     height, area = self.area_reward()
    #     height = np.sum(height)
    #     area = np.sum(area)

    #     for al_id, ego_agent_name in enumerate(self.red_agents):
    #         if not self.red_death[al_id]:
    #             prev_health = self.prev_obs_dict['red'][ego_agent_name]['LifeCurrent']
    #             current_health = self.obs_dict['red'][ego_agent_name]['LifeCurrent']
    #             if int(current_health) == 0 or int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
    #                 self.red_death[al_id] = 1
    #                 if not self.reward_only_positive:
    #                     delta_deaths -= self.reward_death_value * neg_scale 
    #                 delta_ally += neg_scale * prev_health
    #             else:
    #                 delta_ally += neg_scale * (prev_health - current_health)

    #     for e_id, op_agent_name in enumerate(self.blue_agents):
    #         if not self.blue_death[e_id]:
    #             prev_health = self.prev_obs_dict['blue'][op_agent_name]['LifeCurrent']
    #             current_health = self.obs_dict['blue'][op_agent_name]['LifeCurrent']
    #             if int(current_health) == 0 or int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) != 99:
    #                 self.blue_death[e_id] = 1
    #                 if int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) == 0:
    #                     delta_deaths += self.reward_death_value
    #                 else:
    #                     delta_deaths += 2 * self.reward_death_value
    #                 delta_enemy += prev_health
    #             else:
    #                 delta_enemy += prev_health - current_health 

    #     if self.reward_only_positive:
    #         reward = abs(delta_enemy + delta_deaths)
    #     else:
    #         reward = delta_enemy + delta_deaths - delta_ally + locked - (height + area) * neg_scale

    #     return reward

    def reward_single_agent_mode(self):
        pass

    def area_reward(self):

        height = np.zeros((self.num_agents), dtype=np.float32)
        out_area = np.zeros((self.num_agents), dtype=np.float32)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                continue
            if self.obs_dict['red'][ego_agent_name]['position/h-sl-ft'] < self.min_height:
                height[al_id] = 1
            if self.obs_dict['red'][ego_agent_name]['IsOutOfValidBattleArea']:
                out_area[al_id] = 1

        return height, out_area

    def locked_reward(self):

        locked_advantage = np.zeros((self.red_agents_num), dtype=np.float32)
        be_locked_advantage = np.zeros((self.blue_agents_num), dtype=np.float32)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                continue
            if int(self.obs_dict['red'][ego_agent_name]['SRAAMTargetLocked']) != 99 or \
                int(self.obs_dict['red'][ego_agent_name]['AMRAAMlockedTarget']) != 99:
                    locked_advantage[al_id] = 1

        for e_id, op_agent_name in enumerate(self.blue_agents):
            if int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) != 99:
                continue
            if int(self.obs_dict['blue'][op_agent_name]['SRAAMTargetLocked']) != 99 or \
                int(self.obs_dict['blue'][op_agent_name]['AMRAAMlockedTarget']) != 99:
                    be_locked_advantage[e_id] = 1
        
        return locked_advantage, be_locked_advantage

    def get_number_of_agents(self):
        return self.red_agents_num

    def render(self):
        pass

    def latitude_2_xyz(self, longitude, latitude, height):
        PI = 3.1415926
        a = 6378137.0
        e2 = 0.00669438002290
        longitude = longitude * PI / 180
        latitude = latitude * PI / 180

        fac1 = 1 - e2 * math.sin(latitude) * math.sin(latitude)
        N = a / math.sqrt(fac1)
        Daita_h = 0
        h = Daita_h + height

        x = (N + h) * math.cos(latitude) * math.cos(longitude)
        y = (N + h) * math.cos(latitude) * math.sin(longitude)
        z = (N * (1 - e2) + h) * math.sin(latitude)
        return x, y, z

    def look_vector(self, from_vector, to_vector):
        x, y, z = from_vector
        xp, yp, zp = to_vector
        dx, dy, dz = xp - x, yp - y, zp - z
        cos_elevation = (x * dx + y * dy + z * dz) / math.sqrt(
            (x * x + y * y + z * z) * (dx * dx + dy * dy + dz * dz))
        elevation = 90.0 - (math.acos(cos_elevation) * 180.0 / math.pi)
        cos_azimuth = (-z * x * dx - z * y * dy + (x * x + y * y) * dz) / math.sqrt(
            (x * x + y * y) * (x * x + y * y + z * z) * (dx * dx + dy * dy + dz * dz))
        sin_azimuth = (-y * dx + x * dy) / math.sqrt((x * x + y * y) * (dx * dx + dy * dy + dz * dz))
        azimuth = math.atan2(cos_azimuth, sin_azimuth) * 180.0 / math.pi
        if azimuth > 90:
            azimuth = 450 - azimuth
        else:
            azimuth = 90 - azimuth

        return np.array(elevation, dtype=np.float32),  np.array(azimuth, dtype=np.float32)

    def print(self, data):
        if self.port == 8888:
            print(data)

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """

        if self.reward_sparse:
            return 0

        neg_scale = self.reward_negative_scale
        delta_ally = 0
        delta_enemy = 0
        delta_deaths = 0
        weapon_launch = 0
        locked = 0
        area = 0
        height = 0

        # weapon_launch = np.sum(self.invalid_launch) * neg_scale
        locked_adv, be_locked_adv = self.locked_reward()
        locked += np.sum(locked_adv) * 2
        locked -= np.sum(be_locked_adv) * neg_scale

        height, area = self.area_reward()
        height = np.sum(height)
        area = np.sum(area)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if not self.red_death[al_id]:
                prev_health = self.prev_obs_dict['red'][ego_agent_name]['LifeCurrent']
                current_health = self.obs_dict['red'][ego_agent_name]['LifeCurrent']
                if int(current_health) == 0 or int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                    self.red_death[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += neg_scale * prev_health
                else:
                    delta_ally += neg_scale * (prev_health - current_health)

        for e_id, op_agent_name in enumerate(self.blue_agents):
            if not self.blue_death[e_id]:
                prev_health = self.prev_obs_dict['blue'][op_agent_name]['LifeCurrent']
                current_health = self.obs_dict['blue'][op_agent_name]['LifeCurrent']
                if int(current_health) == 0 or int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) != 99:
                    self.blue_death[e_id] = 1
                    if int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) == 0:
                        delta_deaths += self.reward_death_value
                    else:
                        delta_deaths += 2 * self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - current_health

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)
        else:
            reward = delta_enemy + delta_deaths - delta_ally + locked - (height + area) * neg_scale - weapon_launch

        for al_id, ego_agent_name in enumerate(self.red_agents):
            for e_id, op_agent_name in enumerate(self.blue_agents):
                if not self.red_death[al_id] and not self.blue_death[e_id]:
                    th_ang, th_dis = self.one_threat_index(ego_agent_name, op_agent_name)
                    reward += 3 * (1. - th_ang) + 2 * (1. - th_dis)
        return reward

    def one_threat_index(self, ego_agent_name, op_agent_name):
        try:

            ego_x, ego_y, ego_z = self.GPS_to_xyz(self.obs_dict['red'][ego_agent_name]['position/lat-geod-deg'],
                                               self.obs_dict['red'][ego_agent_name]['position/long-gc-deg'],
                                               self.obs_dict['red'][ego_agent_name]['position/h-sl-ft'])

            op_x, op_y, op_z = self.GPS_to_xyz(self.obs_dict['blue'][op_agent_name]['position/lat-geod-deg'],
                                            self.obs_dict['blue'][op_agent_name]['position/long-gc-deg'],
                                            self.obs_dict['blue'][op_agent_name]['position/h-sl-ft'])

            ego_x_ang, ego_y_ang, ego_z_ang = self.Euler_to_xyz(self.obs_dict['red'][ego_agent_name]['attitude/psi-deg'],
                                                           self.obs_dict['red'][ego_agent_name]['attitude/pitch-rad'],
                                                           self.obs_dict['red'][ego_agent_name]['attitude/roll-rad'])

            op_x_ang, op_y_ang, op_z_ang = self.Euler_to_xyz(self.obs_dict['blue'][op_agent_name]['attitude/psi-deg'],
                                                        self.obs_dict['blue'][op_agent_name]['attitude/pitch-rad'],
                                                        self.obs_dict['blue'][op_agent_name]['attitude/roll-rad'])

            # 距离威胁指数
            dis_attack = 40000.
            dis_render = 60000.
            dis_blind = 5000.
            dis_air = np.linalg.norm([ego_x - op_x, ego_y - op_y, ego_z - op_z])
            if dis_air > dis_render:
                th_dis = 0.01
            elif dis_air <= dis_render and dis_render >= dis_attack:
                th_dis = 0.01 + (dis_air - dis_render) * (0.4 - 0.01) / (dis_attack - dis_render)
            elif dis_air <= dis_attack and dis_render >= dis_blind:
                th_dis = 0.4 * dis_attack / dis_air
            else:
                th_dis = 0.1
            if '99' not in str(self.obs_dict['red'][ego_agent_name]['AMRAAMlockedTarget']):
                th_dis *= 0.4

            # 角度威胁指数
            ego_array = np.array([ego_x_ang, ego_y_ang, ego_z_ang])
            pos_arrray = np.array([op_x - ego_x, op_y - ego_y, op_z - ego_z])
            op_array = np.array([op_x_ang, op_y_ang, op_z_ang])

            theta_ang = self.array_angle(ego_array, pos_arrray)
            phi_ang = self.array_angle(op_array, pos_arrray)
            w_angle = 80000. / (dis_air + 100000.)  # 威胁角权重，当距离远时仅关注进攻角
            th_ang = ((1 - w_angle) * theta_ang + w_angle * phi_ang) / 360.
        except KeyError:
            th_ang, th_dis = 0., 0.
        return th_ang, th_dis

    def array_angle(self, x, y):  # 计算两向量夹角
        return np.arccos(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))) * 180. / np.pi

    def into_view(self, attribute):
        print(f"attribute: {attribute}")
        fly_ids = np.zeros((5), dtype=np.float32)
        index = int(attribute)
        st = 0
        while index:
            fly = index % 10
            index = index // 10
            fly_ids[st] = fly
            st += 1

        return fly_ids

    def GPS_to_xyz(self, lat, lon, height):
        # 输入经纬度、海拔（ft）
        # 输入参考系下坐标值XYZ (m) X:North Y:East Z:DOWN

        # CONSTANTS_RADIUS_OF_EARTH = 6356752.  # 极半径
        # CONSTANTS_RADIUS_OF_EARTH = 6378137.  # 赤道半径
        CONSTANTS_RADIUS_OF_EARTH = 6371000.  # 平均半径

        ft_to_m = 0.3048

        # 可以预先定义参考点，否则取第一个测试点为参考点
        try:
            if not self.ref_lat:
                self.ref_lat = lat
        except AttributeError:
            self.ref_lat = lat

        try:
            if not self.ref_lon:
                self.ref_lon = lon
        except AttributeError:
            self.ref_lon = lon

        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(self.ref_lat)
        ref_lon_rad = math.radians(self.ref_lon)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        ref_sin_lat = math.sin(ref_lat_rad)
        ref_cos_lat = math.cos(ref_lat_rad)

        cos_d_lon = math.cos(lon_rad - ref_lon_rad)

        arg = np.clip(ref_sin_lat * sin_lat + ref_cos_lat * cos_lat * cos_d_lon, -1.0, 1.0)
        c = math.acos(arg)

        k = 1.0
        if abs(c) > 0:
            k = (c / math.sin(c))

        x = float(k * (ref_cos_lat * sin_lat - ref_sin_lat * cos_lat * cos_d_lon) * CONSTANTS_RADIUS_OF_EARTH)
        y = float(k * cos_lat * math.sin(lon_rad - ref_lon_rad) * CONSTANTS_RADIUS_OF_EARTH)
        z = float(0. - ft_to_m * height)

        return x, y, z

    def Euler_to_xyz(self, yaw, pitch, roll):
        # 注意输入顺序和单位  :  yaw-航向角-角度    pitch-俯仰角-弧度    roll-翻滚角-弧度
        # 飞机朝向仅与偏航角、俯仰角有关 与翻滚角无关
        x = np.cos(yaw * np.pi / 180.)
        y = np.sin(yaw * np.pi / 180.)
        z = -np.sin(pitch) + 0. * roll
        return x, y, z
