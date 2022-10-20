import os
import os.path as osp
import sys

parent_path = osp.dirname(__file__)
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
        self.single_agent_mode = kwargs.get("single_agent_mode", False)
        self.win_record = deque(maxlen=30)

        self.min_height = kwargs.get("min_height", 26000)
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
        self.max_reward = self.reward_win +\
                            2 * self.reward_death_value * self.blue_agents_num+\
                               200*self.blue_agents_num +\
                           self.blue_agents_num +\
                           self.red_agents_num +\
                           self.red_agents_num*self.blue_agents_num*5

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

        if self.red_agents_num == 1:
            shape = 127
        elif self.red_agents_num == 5:
            shape = 499
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
        self.cum_rewards = 0
        self._episode_steps = 0
        self.record = False
        self.red_weapon_state = dict()
        for ego_agent_name in self.red_agents:
            self.red_weapon_state[ego_agent_name] = {}
        # setting = np.random.randint(0, 3)
        setting = self._episode_steps % 2
        if init:
            # print(self.dict_init)
            self.obs_dict = self.env.reset(self.dict_init[setting])
            self.prev_obs_dict = None
        else:
            self.obs_dict = self.env.reset(self.dict_reset[setting])
            self.prev_obs_dict = None
        obs, obs_op = self.get_obs()
        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] =obs, obs_op

        return obs_dict

    def weapon_actions_(self, ego_actions, camp='red'):
        return ego_actions

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
            if int(self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange']) == 0:
                continue
            fly_ids = self.into_view(self.prev_obs_dict[camp][agent_name]['TargetEnterAttackRange'])
            goals = fly_ids.nonzero()[0]
            if len(goals) == 0:
                continue
            ego_x, ego_y, ego_z = self.GPS_to_xyz(self.prev_obs_dict[camp][agent_name]['position/lat-geod-deg'],
                                                  self.prev_obs_dict[camp][agent_name]['position/long-gc-deg'],
                                                  self.prev_obs_dict[camp][agent_name]['position/h-sl-ft'])
            srmissile_nums = self.prev_obs_dict[camp][agent_name]['SRAAMCurrentNum']
            armissile_nums = self.prev_obs_dict[camp][agent_name]['AMRAAMCurrentNum']
            bullet_nums = self.prev_obs_dict[camp][agent_name]['BulletCurrentNum']
            aim_mode = self.prev_obs_dict[camp][agent_name]['AimMode']
            sr_locked = int(self.prev_obs_dict[camp][agent_name]['SRAAMTargetLocked'])
            ar_locked = int(self.prev_obs_dict[camp][agent_name]['AMRAAMlockedTarget'])
            if len(goals) == 1:
                item = goals.item()
                enemy = 'blue_'+str(item) if camp_another == 'blue' else 'red_'+str(item)
                if self.prev_obs_dict[camp_another][enemy]['DeathEvent'] != 99:
                    continue
                enemy_x, enemy_y,enemy_z = self.GPS_to_xyz(
                    self.prev_obs_dict[camp_another][enemy]['position/lat-geod-deg'],
                    self.prev_obs_dict[camp_another][enemy]['position/long-gc-deg'],
                    self.prev_obs_dict[camp_another][enemy]['position/h-sl-ft'])
                dist = np.linalg.norm(np.array([enemy_x-ego_x, enemy_y-ego_y, enemy_z-ego_z]))

                if dist < 2000:
                    if bullet_nums > 0:
                     ego_actions[al_id][1] = 2
                    else:
                        continue
                elif dist <= 12000:
                    if aim_mode == 0:
                        if srmissile_nums != 0:
                            if sr_locked != 9:
                                ego_actions[al_id][1] = 1
                        elif armissile_nums != 0:
                            ego_actions[al_id][0] = 1
                    else:
                        if srmissile_nums == 0:
                            if armissile_nums != 0 and ar_locked != 9999:
                                ego_actions[al_id][1] = 1
                        else:
                            ego_actions[al_id][0] = 1
                else:
                    if aim_mode == 1:
                        if armissile_nums != 0 and ar_locked != 9999:
                            ego_actions[al_id][1] = 1
                    else:
                        ego_actions[al_id][0] = 1

            else: # goals > 1
                # if int(sr_locked) == 9 and int(ar_locked) == 9999:
                #     continue
                if aim_mode == 0:
                    if armissile_nums > 0:
                        ego_actions[al_id][0] = 1
                    elif srmissile_nums > 0:
                        if sr_locked != 9:
                            enemy = 'blue_'+str(sr_locked) if camp_another == 'blue' else 'red_'+str(sr_locked)
                            if self.prev_obs_dict[camp_another][enemy]['DeathEvent'] != 99:
                                continue
                            enemy_x, enemy_y,enemy_z = self.GPS_to_xyz(
                                self.prev_obs_dict[camp_another][enemy]['position/lat-geod-deg'],
                                self.prev_obs_dict[camp_another][enemy]['position/long-gc-deg'],
                                self.prev_obs_dict[camp_another][enemy]['position/h-sl-ft'])
                            dist = np.linalg.norm(np.array([enemy_x-ego_x, enemy_y-ego_y, enemy_z-ego_z]))
                            if dist <= 12000:
                                ego_actions[al_id][1] = 1
                elif aim_mode == 1:
                    if armissile_nums > 0:
                        if ar_locked != 9999:
                            ego_actions[al_id][1] = 1
                    elif srmissile_nums > 0:
                        ego_actions[al_id][0] = 1

        return ego_actions

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

        # ego_weapon_actions = self.weapon_actions_(ego_weapon_actions, camp='red')
        # op_weapon_actions = self.weapon_actions_(op_weapon_actions, camp='blue')

        infos = {}
        dones = np.zeros(self.num_agents, dtype=bool)

        self.process_actions(ego_action, op_action, ego_weapon_actions, op_weapon_actions)

        self.obs_dict = self.env.step(self.current_actions)
        self._episode_steps += 1

        if self.single_agent_mode:
            reward = self.reward_single_agent_mode()
        else:
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
        infos['win'] = np.zeros((self.red_agents_num), dtype=np.float32)
        infos['lose'] = np.zeros((self.red_agents_num), dtype=np.float32)
        infos['draw'] = np.zeros((self.red_agents_num), dtype=np.float32)

        if self.red_all_dead:
            if self.reward_sparse and not self.single_agent_mode:
                reward = -1
            else:
                reward += self.reward_defeat
            infos['lose'] = np.ones((self.red_agents_num), dtype=np.float32)
        elif self.red_all_dead and self.blue_all_dead:
            infos['draw'] = np.ones((self.red_agents_num), dtype=np.float32)
        if self._episode_steps >= self.episode_limit:
            if self.blue_all_dead and not self.red_all_dead:
                if self.reward_sparse and not self.single_agent_mode:
                    reward = 1
                else:
                    reward += self.reward_win
                infos['win'] = np.ones((self.red_agents_num), dtype=np.float32)
            else:
                infos['draw'] = np.ones((self.red_agents_num), dtype=np.float32)
                self.record = True
        else:
            infos['win'] = np.zeros((self.red_agents_num), dtype=np.float32)
            infos['lose'] = np.zeros((self.red_agents_num), dtype=np.float32)
            infos['draw'] = np.zeros((self.red_agents_num), dtype=np.float32)

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

    def get_obs_op(self, ego_x, ego_y, ego_z,
                   ego_ang_x, ego_ang_y, ego_ang_z,
                   camp='blue'):
        """
        opponent info including health and vel infos.
        """

        enemy_agents = self.camp_2_agents[camp]
        enemy_agents_num = self.camp_2_agents_num[camp]
        enemy_infos = np.zeros((enemy_agents_num, 49), dtype=np.float32)

        for i, name in enumerate(enemy_agents):
            if int(self.obs_dict[camp][name]['DeathEvent']) != 99:
                continue
            enemy_infos[i, 0] = self.obs_dict[camp][name]['LifeCurrent']
            enemy_infos[i, 1] = self.obs_dict[camp][name]['position/long-gc-deg']
            enemy_infos[i, 2] = self.obs_dict[camp][name]['position/lat-geod-deg']
            enemy_infos[i, 3] = self.obs_dict[camp][name]['position/h-sl-ft']
            enemy_x, enemy_y, enemy_z = self.GPS_to_xyz(enemy_infos[i, 2], enemy_infos[i, 1],
                                                        enemy_infos[i, 3])
            enemy_ang_x, enemy_ang_y, enemy_ang_z = \
                self.Euler_to_xyz(self.obs_dict[camp][name]['attitude/psi-deg'],
                                  self.obs_dict[camp][name]['attitude/pitch-rad'],
                                  self.obs_dict[camp][name]['attitude/roll-rad'])

            ego_direction_vector = np.array([ego_ang_x, ego_ang_y, ego_ang_z])
            op_direction_vector = np.array([enemy_ang_x, enemy_ang_y, enemy_ang_z])
            position_direction_vector = np.array([enemy_ang_x - ego_x, enemy_ang_y - ego_y, enemy_ang_z - ego_z])

            theta_angle = self.calculation_tool(ego_direction_vector, position_direction_vector, mode='array_angle')
            phi_angle = self.calculation_tool(op_direction_vector, position_direction_vector, mode='array_angle')

            relative_x = enemy_x - ego_x
            relative_y = enemy_y - ego_y
            relative_z = enemy_z - ego_z
            relative_dist = np.linalg.norm([relative_x, relative_y, relative_z])

            enemy_infos[i, 4] = relative_x
            enemy_infos[i, 5] = relative_y
            enemy_infos[i, 6] = relative_z
            enemy_infos[i, 7] = relative_dist
            enemy_infos[i, 8] = theta_angle
            enemy_infos[i, 9] = phi_angle
            enemy_infos[i, 10] = enemy_ang_x
            enemy_infos[i, 11] = enemy_ang_y
            enemy_infos[i, 12] = enemy_ang_z

            enemy_infos[i, 13] = self.obs_dict[camp][name]['attitude/psi-deg']
            enemy_infos[i, 14] = self.obs_dict[camp][name]['velocities/ve-fps']
            enemy_infos[i, 15] = self.obs_dict[camp][name]['velocities/h-dot-fps']
            enemy_infos[i, 16] = self.obs_dict[camp][name]['velocities/v-north-fps']
            enemy_infos[i, 17] = self.obs_dict[camp][name]['velocities/v-east-fps']
            enemy_infos[i, 18] = self.obs_dict[camp][name]['velocities/v-down-fps']

            enemy_infos[i, 19] = self.obs_dict[camp][name]['SRAAMCurrentNum']
            enemy_infos[i, 20] = self.obs_dict[camp][name]['AMRAAMCurrentNum']
            enemy_infos[i, 21] = self.obs_dict[camp][name]['BulletCurrentNum']
            enemy_infos[i, 22+int(self.obs_dict[camp][name]['IfPresenceHitting'])] = 1

            if int(self.obs_dict[camp][name]['TargetIntoView']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['TargetIntoView'])
                enemy_infos[i, 24:29] = fly_ids
            if int(self.obs_dict[camp][name]['AllyIntoView']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['AllyIntoView'])
                enemy_infos[i, 29:34] = fly_ids
            if int(self.obs_dict[camp][name]['TargetEnterAttackRange']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['TargetEnterAttackRange'])
                enemy_infos[i, 34:39] = fly_ids
            if int(self.obs_dict[camp][name]['SRAAMTargetLocked']) != 9:
                fly_ids = int(self.obs_dict[camp][name]['SRAAMTargetLocked'])
                enemy_infos[i, 39+fly_ids] = 1
            if int(self.obs_dict[camp][name]['AMRAAMlockedTarget']) != 9999:
                fly_list = self.lockedtarget(self.obs_dict[camp][name]['AMRAAMlockedTarget'])
                enemy_infos[i, 44+fly_list] = 1

        return enemy_infos.flatten()

    def get_obs_alley(self, ego_agent_name,
                      ego_x, ego_y, ego_z, camp='red'):

        alley_agents = deepcopy(self.camp_2_agents[camp])
        alley_agents.pop(int(ego_agent_name.split('_')[-1]))
        alley_agent_num = len(alley_agents)
        alley_infos = np.zeros((alley_agent_num, 43), dtype=np.float32)

        for i, name in enumerate(alley_agents):
            if name == ego_agent_name:
                continue
            if int(self.obs_dict[camp][name]['DeathEvent']) != 99:
                continue
            alley_infos[i, 0] = self.obs_dict[camp][name]['LifeCurrent']
            alley_infos[i, 1] = self.obs_dict[camp][name]['position/long-gc-deg']
            alley_infos[i, 2] = self.obs_dict[camp][name]['position/lat-geod-deg']
            alley_infos[i, 3] = self.obs_dict[camp][name]['position/h-sl-ft']
            alley_x, alley_y, alley_z = self.GPS_to_xyz( alley_infos[i, 2], alley_infos[i, 1],
                                                        alley_infos[i, 3])

            relative_x = alley_x - ego_x
            relative_y = alley_y - ego_y
            relative_z = alley_z - ego_z
            relative_dist = np.linalg.norm([relative_x, relative_y, relative_z])
            alley_infos[i, 4] = relative_x
            alley_infos[i, 5] = relative_y
            alley_infos[i, 6] = relative_z
            alley_infos[i, 7] = relative_dist

            alley_infos[i, 8] = self.obs_dict[camp][name]['attitude/psi-deg']
            alley_infos[i, 9] = self.obs_dict[camp][name]['velocities/ve-fps']
            alley_infos[i, 10] = self.obs_dict[camp][name]['velocities/h-dot-fps']
            alley_infos[i, 11] = self.obs_dict[camp][name]['velocities/v-north-fps']
            alley_infos[i, 12] = self.obs_dict[camp][name]['velocities/v-east-fps']
            alley_infos[i, 13] = self.obs_dict[camp][name]['velocities/v-down-fps']

            alley_infos[i, 14] = self.obs_dict[camp][name]['SRAAMCurrentNum']
            alley_infos[i, 15] = self.obs_dict[camp][name]['AMRAAMCurrentNum']
            alley_infos[i, 16+int(self.obs_dict[camp][name]['IfPresenceHitting'])] = 1

            if int(self.obs_dict[camp][name]['TargetIntoView']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['TargetIntoView'])
                alley_infos[i, 18:23] = fly_ids
            if int(self.obs_dict[camp][name]['AllyIntoView']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['AllyIntoView'])
                alley_infos[i, 23:28] = fly_ids
            if int(self.obs_dict[camp][name]['TargetEnterAttackRange']) != 0:
                fly_ids = self.into_view(self.obs_dict[camp][name]['TargetEnterAttackRange'])
                alley_infos[i, 28:33] = fly_ids
            if int(self.obs_dict[camp][name]['SRAAMTargetLocked']) != 9:
                fly_ids = int(self.obs_dict[camp][name]['SRAAMTargetLocked'])
                alley_infos[i, 33+fly_ids] = 1
            if int(self.obs_dict[camp][name]['AMRAAMlockedTarget']) != 9999:
                fly_list = self.lockedtarget(self.obs_dict[camp][name]['AMRAAMlockedTarget'])
                alley_infos[i, 38+fly_list] = 1

        return alley_infos.flatten()

    def get_obs_agent(self, agent_name, camp='red'):
        """Returns observation for agent_name. The observation is composed of:

        - ego agent features
        - ally features
        - enemy features
        - ally features
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """

        ego_infos = np.zeros((68), dtype=np.float32)

        ego_infos[0] = self.obs_dict[camp][agent_name]['LifeCurrent']
        ego_infos[1] = self.obs_dict[camp][agent_name]['position/long-gc-deg']
        ego_infos[2] = self.obs_dict[camp][agent_name]['position/lat-geod-deg']
        ego_infos[3] = self.obs_dict[camp][agent_name]['position/h-sl-ft']
        # ego agent x y z
        ego_x, ego_y, ego_z = self.GPS_to_xyz(lat=self.obs_dict[camp][agent_name]['position/lat-geod-deg'],
                                  lon=self.obs_dict[camp][agent_name]['position/long-gc-deg'],
                                  height=self.obs_dict[camp][agent_name]['position/h-sl-ft'])

        ego_ang_x, ego_ang_y, ego_ang_z = \
            self.Euler_to_xyz(self.obs_dict[camp][agent_name]['attitude/psi-deg'],
                                self.obs_dict[camp][agent_name]['attitude/pitch-rad'],
                                self.obs_dict[camp][agent_name]['attitude/roll-rad'])

        ego_infos[4] = ego_x
        ego_infos[5] = ego_y
        ego_infos[6] = ego_z
        ego_infos[7] = ego_ang_x
        ego_infos[8] = ego_ang_y
        ego_infos[9] = ego_ang_z

        ego_infos[10] = self.obs_dict[camp][agent_name]['attitude/pitch-rad']
        ego_infos[11] = self.obs_dict[camp][agent_name]['attitude/roll-rad']
        ego_infos[12] = self.obs_dict[camp][agent_name]['attitude/psi-deg']
        ego_infos[13] = self.obs_dict[camp][agent_name]['velocities/u-fps']
        ego_infos[14] = self.obs_dict[camp][agent_name]['velocities/v-fps']
        ego_infos[15] = self.obs_dict[camp][agent_name]['velocities/w-fps']

        ego_infos[16] = self.obs_dict[camp][agent_name]['velocities/v-north-fps']
        ego_infos[17] = self.obs_dict[camp][agent_name]['velocities/v-east-fps']
        ego_infos[18] = self.obs_dict[camp][agent_name]['velocities/v-down-fps']

        ego_infos[19] = self.obs_dict[camp][agent_name]['velocities/p-rad_sec']
        ego_infos[20] = self.obs_dict[camp][agent_name]['velocities/q-rad_sec']
        ego_infos[21] = self.obs_dict[camp][agent_name]['velocities/ve-fps']
        ego_infos[22] = self.obs_dict[camp][agent_name]['velocities/h-dot-fps']
        ego_infos[23] = self.obs_dict[camp][agent_name]['forces/load-factor']
        ego_infos[24] = self.obs_dict[camp][agent_name]['IsOutOfValidBattleArea']
        ego_infos[25] = self.obs_dict[camp][agent_name]['OutOfValidBattleAreaCurrentDuration']
        ego_infos[26] = self.obs_dict[camp][agent_name]['SRAAMCurrentNum']
        ego_infos[27] = self.obs_dict[camp][agent_name]['AMRAAMCurrentNum']
        ego_infos[28] = self.obs_dict[camp][agent_name]['BulletCurrentNum']
        ego_infos[29+int(self.obs_dict[camp][agent_name]['AimMode'])] = 1
        ego_infos[31+int(self.obs_dict[camp][agent_name]['SRAAM1_CanReload'])] = 1
        ego_infos[33+int(self.obs_dict[camp][agent_name]['SRAAM2_CanReload'])] = 1
        ego_infos[35+int(self.obs_dict[camp][agent_name]['AMRAAMCanReload'])] = 1
        ego_infos[37+int(self.obs_dict[camp][agent_name]['IfPresenceHitting'])] = 1

        if int(self.obs_dict[camp][agent_name]['TargetIntoView']) != 0:
            fly_ids = self.into_view(self.obs_dict[camp][agent_name]['TargetIntoView'])
            ego_infos[39:44] = fly_ids
        if int(self.obs_dict[camp][agent_name]['AllyIntoView']) != 0:
            fly_ids = self.into_view(self.obs_dict[camp][agent_name]['AllyIntoView'])
            ego_infos[44:49] = fly_ids
        if int(self.obs_dict[camp][agent_name]['TargetEnterAttackRange']) != 0:
            fly_ids = self.into_view(self.obs_dict[camp][agent_name]['TargetEnterAttackRange'])
            ego_infos[49:54] = fly_ids

        if int(self.obs_dict[camp][agent_name]['SRAAMTargetLocked']) != 9:
            fly_ids = int(self.obs_dict[camp][agent_name]['SRAAMTargetLocked'])
            ego_infos[54+fly_ids] = 1
        if int(self.obs_dict[camp][agent_name]['AMRAAMlockedTarget']) != 9999:
            fly_list = self.lockedtarget(self.obs_dict[camp][agent_name]['AMRAAMlockedTarget'])
            ego_infos[59+fly_list] = 1

        ego_infos[64+int(self.obs_dict[camp][agent_name]['MissileAlert'])] = 1
        ego_infos[66] = self.obs_dict[camp][agent_name]['EnvelopeMin']
        ego_infos[67] = self.obs_dict[camp][agent_name]['EnvelopeMax']

        ctrl_state_feats = np.zeros((9), dtype=np.float32)
        ctrl_state_feats[0] = self.obs_dict[camp][agent_name]['fcs/left-aileron-pos-norm']
        ctrl_state_feats[1] = self.obs_dict[camp][agent_name]['fcs/right-aileron-pos-norm']
        ctrl_state_feats[2] = self.obs_dict[camp][agent_name]['fcs/elevator-pos-norm']
        ctrl_state_feats[3] = self.obs_dict[camp][agent_name]['fcs/throttle-pos-norm']

        # last step action
        ctrl_state_feats[4] = self.obs_dict[camp][agent_name]['fcs/aileron-cmd-norm']
        ctrl_state_feats[5] = self.obs_dict[camp][agent_name]['fcs/elevator-cmd-norm']
        ctrl_state_feats[6] = self.obs_dict[camp][agent_name]['fcs/throttle-cmd-norm']
        ctrl_state_feats[7] = self.obs_dict[camp][agent_name]['missile-launch']
        ctrl_state_feats[8] = self.obs_dict[camp][agent_name]['switch-missile']

        alley_feats = self.get_obs_alley(ego_agent_name=agent_name,
                                         ego_x=ego_x, ego_y=ego_y, ego_z=ego_z, camp=camp)

        if camp == 'red':
            enemy_feats = self.get_obs_op(ego_x=ego_x, ego_y=ego_y, ego_z=ego_z,
                                          ego_ang_x=ego_ang_x, ego_ang_y=ego_ang_y, ego_ang_z=ego_ang_z,
                                          camp='blue')
            agent_id_feats = np.zeros(self.red_agents_num, dtype=np.float32)
            # threat info
            # threat_info = self._oracal_guiding_feature(agent_name, camp, op_camp='blue')
        else:
            enemy_feats = self.get_obs_op(ego_x=ego_x, ego_y=ego_y, ego_z=ego_z,
                                          ego_ang_x=ego_ang_x, ego_ang_y=ego_ang_y, ego_ang_z=ego_ang_z,
                                          camp='red')
            agent_id_feats = np.zeros(self.blue_agents_num, dtype=np.float32)
            # threat info
            # threat_info = self._oracal_guiding_feature(agent_name, camp, op_camp='red')

        if self.apply_agent_ids:
            if camp == 'red':
                agent_id_feats[self.red_ni_mapping[agent_name]] = 1
            else:
                agent_id_feats[self.blue_ni_mapping[agent_name]] = 1

        obs_all = np.concatenate([
            ego_infos.flatten(), ctrl_state_feats.flatten(), alley_feats.flatten(),
            enemy_feats.flatten(), agent_id_feats.flatten(),
            # threat_info.flatten(), agent_id_feats.flatten()
        ])
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

    def into_view(self, attribute):
        fly_ids = np.zeros((5), dtype=np.float32)
        index = int(attribute)
        st = 0
        while index:
            fly = index % 10
            index = index // 10
            fly_ids[st] = fly
            st += 1

        return fly_ids

    def lockedtarget(self, attribute):
        index = int(attribute)
        st = 0
        fly_ids = []
        while index:
            fly = index % 10
            index = index // 10
            if fly != 9:
                fly_ids.append(fly)
            st += 1
        if st != 4 and 0 not in fly_ids:
            fly_ids.append(0)

        return np.array(fly_ids)

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
        if np.sum(self.blue_death) == self.blue_agents_num:
            self.blue_all_dead = True

        if np.sum(self.red_death_missile) == self.red_agents_num:
            self.red_all_dead_missile = True
        if np.sum(self.blue_death_missile) == self.blue_agents_num:
            self.blue_all_dead_missile = True

        return self.red_all_dead

    def seed(self, seed=1):
        random.seed(seed)
        np.random.seed(seed=seed)

    def reward_single_agent_mode(self):
        if self.reward_sparse:
            return 0

        neg_scale = self.reward_negative_scale
        delta_ally = 0
        delta_enemy = 0
        delta_deaths = 0
        locked = 0
        area = 0
        height = 0

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
                    delta_ally += neg_scale * prev_health
                else:
                    delta_ally += neg_scale * (prev_health - current_health)

        reward = - delta_ally + locked - (height + area) * neg_scale
        return reward

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

    def alert_reward(self):
        ego_alert = np.ones((self.red_agents_num), dtype=np.float32)
        for al_id, ego_agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                ego_alert[al_id] = 0
                continue
            if int(self.obs_dict['red'][ego_agent_name]['MissileAlert']) == 1:
                    ego_alert[al_id] = 0
        return ego_alert

    def locked_reward(self):

        locked_advantage = np.zeros((self.red_agents_num), dtype=np.float32)
        be_locked_advantage = np.zeros((self.blue_agents_num), dtype=np.float32)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                continue
            if int(self.obs_dict['red'][ego_agent_name]['SRAAMTargetLocked']) != 9 or \
                int(self.obs_dict['red'][ego_agent_name]['AMRAAMlockedTarget']) != 9999:
                    locked_advantage[al_id] = 1

        for e_id, op_agent_name in enumerate(self.blue_agents):
            if int(self.obs_dict['blue'][op_agent_name]['DeathEvent']) != 99:
                continue
            if int(self.obs_dict['blue'][op_agent_name]['SRAAMTargetLocked']) != 9 or \
                int(self.obs_dict['blue'][op_agent_name]['AMRAAMlockedTarget']) != 9999:
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
        locked = 0
        area = 0
        height = 0
        missile_alert = 0

        locked_adv, be_locked_adv = self.locked_reward()
        ego_alert = self.alert_reward()
        locked += np.sum(locked_adv) * 3
        locked -= np.sum(be_locked_adv) * neg_scale
        missile_alert += np.sum(ego_alert)

        height, area = self.area_reward()
        height = np.sum(height)
        area = np.sum(area)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if not self.red_death[al_id]:
                prev_health = self.prev_obs_dict['red'][ego_agent_name]['LifeCurrent']
                current_health = self.obs_dict['red'][ego_agent_name]['LifeCurrent']
                if int(current_health) == 0 or int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                    self.red_death[al_id] = 1
                    delta_deaths -= self.reward_death_value
                    delta_ally += prev_health
                else:
                    delta_ally += (prev_health - current_health)

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

        reward = delta_enemy + delta_deaths + missile_alert - delta_ally + locked - height - area
        
        step_reward = 0.  # 最大立即回报 = 红方飞机数量 * 5
        for al_id, ego_agent_name in enumerate(self.red_agents):
            agent_reward_list = []
            distance_list = []
            for e_id, op_agent_name in enumerate(self.blue_agents):
                if not self.red_death[al_id] and not self.blue_death[e_id]:
                    threat_dict = self.one_threat_index(ego_agent_name, op_agent_name)
                    agent_reward_list.append(3 * (1. - threat_dict['threat_ang']) +
                                             2 * (1. - threat_dict['threat_dis']))
                    distance_list.append(threat_dict['distance'])
            w_list = self.calculation_tool(distance_list, mode='distance_weight')
            step_reward += np.dot(w_list, agent_reward_list)
        reward += step_reward
        return reward

    def one_threat_index(self, ego_agent_name, op_agent_name):  # 输入 红蓝双方各一架飞机名字，输出 角度威胁和速度威胁
        try:
            ego_x, ego_y, ego_z = \
                self.GPS_to_xyz(self.obs_dict['red'][ego_agent_name]['position/lat-geod-deg'],
                                self.obs_dict['red'][ego_agent_name]['position/long-gc-deg'],
                                self.obs_dict['red'][ego_agent_name]['position/h-sl-ft'])

            op_x, op_y, op_z = \
                self.GPS_to_xyz(self.obs_dict['blue'][op_agent_name]['position/lat-geod-deg'],
                                self.obs_dict['blue'][op_agent_name]['position/long-gc-deg'],
                                self.obs_dict['blue'][op_agent_name]['position/h-sl-ft'])

            ego_ang_x, ego_ang_y, ego_ang_z = \
                self.Euler_to_xyz(self.obs_dict['red'][ego_agent_name]['attitude/psi-deg'],
                                  self.obs_dict['red'][ego_agent_name]['attitude/pitch-rad'],
                                  self.obs_dict['red'][ego_agent_name]['attitude/roll-rad'])

            op_ang_x, op_ang_y, op_ang_z = \
                self.Euler_to_xyz(self.obs_dict['blue'][op_agent_name]['attitude/psi-deg'],
                                  self.obs_dict['blue'][op_agent_name]['attitude/pitch-rad'],
                                  self.obs_dict['blue'][op_agent_name]['attitude/roll-rad'])

            pitch_reward = 0.

            # 距离威胁
            f_param_AMRAAM_approach = self.calculation_tool([2000., 1.], [35000., 0.4], mode='function_parameters')
            f_param_AMRAAM_avoid = self.calculation_tool([2000., 8.], [40000., 0.4], mode='function_parameters')
            f_param_SRAAM_approach = self.calculation_tool([2000., 1.], [8000., 0.4], mode='function_parameters')
            f_param_SRAAM_avoid = self.calculation_tool([2000., 8.], [10000., 0.4], mode='function_parameters')
            # function_param mode : 'array_angle','function_parameters', 'distance_threat'
            if self._episode_steps <= 1:

                # try , except 此处用于初始化self.red_weapon_state， 实际上该段代码应该在reset()函数实现
                self.red_weapon_state[ego_agent_name]['SRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['AMRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['func_param'] = f_param_AMRAAM_approach
                self.red_weapon_state[ego_agent_name]['threat_time_count'] = 0.
                self.red_weapon_state[ego_agent_name]['th_mode'] = 0

            if self.red_weapon_state[ego_agent_name]['AMRAAMCurrentNum'] != self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum']\
                    and self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum'] != 0:
                self.red_weapon_state[ego_agent_name]['AMRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['threat_time_count'] = self._episode_steps
                self.red_weapon_state[ego_agent_name]['th_mode'] = 1
            elif self.red_weapon_state[ego_agent_name]['AMRAAMCurrentNum'] != self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum'] \
                    and self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum'] == 0:
                self.red_weapon_state[ego_agent_name]['AMRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['AMRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['threat_time_count'] = self._episode_steps
                self.red_weapon_state[ego_agent_name]['th_mode'] = 2
            elif self.red_weapon_state[ego_agent_name]['SRAAMCurrentNum'] != self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum'] \
                    and self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum'] != 0:
                self.red_weapon_state[ego_agent_name]['SRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['threat_time_count'] = self._episode_steps
                self.red_weapon_state[ego_agent_name]['th_mode'] = 3
            elif self.red_weapon_state[ego_agent_name]['SRAAMCurrentNum'] != self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum'] \
                    and self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum'] == 0:
                self.red_weapon_state[ego_agent_name]['SRAAMCurrentNum'] = self.obs_dict['red'][ego_agent_name]['SRAAMCurrentNum']
                self.red_weapon_state[ego_agent_name]['threat_time_count'] = self._episode_steps
                self.red_weapon_state[ego_agent_name]['th_mode'] = 4

            time_count = self._episode_steps - self.red_weapon_state[ego_agent_name]['threat_time_count']
            if 5 <= time_count < 10:
                if self.red_weapon_state[ego_agent_name]['th_mode'] != 0:
                    pitch = self.obs_dict['red'][ego_agent_name]['attitude/pitch-rad']
                    pitch_reward += -np.sin(pitch)
                if self.red_weapon_state[ego_agent_name]['th_mode'] == 1:
                    self.red_weapon_state[ego_agent_name]['func_param'] = f_param_AMRAAM_avoid
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 2:
                    self.red_weapon_state[ego_agent_name]['func_param'] = f_param_AMRAAM_avoid
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 3:
                    self.red_weapon_state[ego_agent_name]['func_param'] = f_param_SRAAM_avoid
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 4:
                    self.red_weapon_state[ego_agent_name]['func_param'] = f_param_SRAAM_avoid
            elif 10 <= time_count < 15:
                if self.red_weapon_state[ego_agent_name]['th_mode'] == 1:
                    param_beg = f_param_AMRAAM_avoid
                    param_end = f_param_AMRAAM_approach
                    self.red_weapon_state[ego_agent_name]['func_param'] = param_beg + (param_end - param_beg) * float(time_count - 9) / 5.
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 2:
                    param_beg = f_param_AMRAAM_avoid
                    param_end = f_param_SRAAM_approach
                    self.red_weapon_state[ego_agent_name]['func_param'] = param_beg + (param_end - param_beg) * float(time_count - 9) / 5.
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 3:
                    param_beg = f_param_SRAAM_avoid
                    param_end = f_param_SRAAM_approach
                    self.red_weapon_state[ego_agent_name]['func_param'] = param_beg + (param_end - param_beg) * float(time_count - 9) / 5.
                elif self.red_weapon_state[ego_agent_name]['th_mode'] == 4:
                    param_beg = f_param_SRAAM_avoid
                    param_end = f_param_AMRAAM_avoid
                    self.red_weapon_state[ego_agent_name]['func_param'] = param_beg + (param_end - param_beg) * float(time_count - 9) / 5.

            dis_render = 60000.
            dis_blind = 2000.
            dis_air = np.linalg.norm([ego_x - op_x, ego_y - op_y, ego_z - op_z])

            if dis_air > dis_render:
                th_dis = 1. - (100000. - dis_air) * (1 - 0.01) / (100000. - dis_render)
            elif dis_air <= dis_render and dis_render >= dis_blind:
                th_dis = self.calculation_tool(dis_air, self.red_weapon_state[ego_agent_name]['func_param'], mode='distance_threat')
            else:
                th_dis = 0.2
            if '99' not in str(self.obs_dict['red'][ego_agent_name]['AMRAAMlockedTarget']) and \
                    (self.red_weapon_state[ego_agent_name]['th_mode'] == 0 or time_count >= 15 or time_count <= 5):
                th_dis *= 0.4

            # 角度威胁
            ego_direction_vector = np.array([ego_ang_x, ego_ang_y, ego_ang_z])
            op_direction_vector = np.array([op_ang_x, op_ang_y, op_ang_z])
            position_direction_vector = np.array([op_x - ego_x, op_y - ego_y, op_z - ego_z])

            theta_angle = self.calculation_tool(ego_direction_vector, position_direction_vector, mode='array_angle')
            phi_angle = self.calculation_tool(op_direction_vector, position_direction_vector, mode='array_angle')

            weight_angle = 70000. / (dis_air + 100000.)  # 威胁角权重，当距离远时仅关注进攻角
            th_ang = ((1 - weight_angle) * theta_angle + weight_angle * phi_angle) / 360.
        except KeyError:
            th_ang, th_dis, pitch_reward, dis_air = 0., 0., 0., 0.
        return {'threat_ang': th_ang, 'threat_dis': th_dis, 'pitch_reward': pitch_reward, 'distance': dis_air}

    def calculation_tool(self, x=None, y=None, mode='array_angle'):  # 计算工具
        if mode == 'array_angle':  # 输入两向量， 计算两向量夹角
            return np.arccos(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))) * 180. / np.pi
        elif mode == 'function_parameters':  # 输入两个点，计算一个经过两点的光滑曲线
            a = (x[0] * x[1] - y[0] * y[1]) / (y[1] - x[1])
            b = x[1] * y[1] * (x[0] - y[0]) / (y[1] - x[1])
            return np.array([a, b])
        elif mode == 'distance_threat':  # 输入距离及参数， 计算距离威胁度
            th_d = y[1] / (y[0] + x)
            return th_d if th_d <= 1. else 1.
        elif mode == 'distance_weight':
            w_list = [sum(x) / dis_ for dis_ in x]
            w_sum = sum(w_list)
            w = np.array([wi_ / w_sum for wi_ in w_list])
            return w
        else:
            print('Error in calculation_tool !! ')

    def array_angle(self, x, y):  # 计算两向量夹角
        return np.arccos(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))) * 180. / np.pi

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
        z = float(0. - ft_to_m * height )

        return x, y, z

    def Euler_to_xyz(self, yaw, pitch, roll):
        # 注意输入顺序和单位  :  yaw-航向角-角度    pitch-俯仰角-弧度    roll-翻滚角-弧度
        # 飞机朝向仅与偏航角、俯仰角有关 与翻滚角无关
        x = np.cos(pitch) * np.cos(yaw * np.pi / 180.)
        y = np.cos(pitch) * np.sin(yaw * np.pi / 180.)
        z = -np.sin(pitch) + 0. * roll
        return x, y, z