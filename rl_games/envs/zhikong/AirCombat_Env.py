import enum
import os.path as osp
import sys

parent_path = osp.dirname(__file__)
print(parent_path)
sys.path.append(parent_path)

import math
from copy import deepcopy
from collections import OrderedDict
import time
import random
from ale_py import os
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete
from zhikong import comm_interface
from .util import init_info, obs_feature_list, act_feature_list

action_aileron = np.linspace(-1., 1., 9) # fcs/aileron-cmd-norm
action_elevator = np.linspace(-1., 1., 9) # fcs/elevator-cmd-norm
action_rudder = np.linspace(-1., 1., 9) # fcs/rudder-cmd-norm
action_throttle = np.linspace(0., 1., 5)  # fcs/throttle-cmd-norm
# action_change_target = [0, ]
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
        self.episode_limit =kwargs.get("episode_max_length", 1000)
        self.change_target = kwargs.get("change_target", False)
        self.single_agent_mode = kwargs.get("single_agent_mode", False)
        
        self.min_height = kwargs.get("min_height", 1000)
        # reward
        self.cum_rewards = 0
        self.reward_win = kwargs.get("reward_win", 100)
        self.reward_sparse = kwargs.get("reward_sparse", False)
        self.reward_negative_scale = kwargs.get("reward_negative_scale", 0.5)
        self.reward_death_value = kwargs.get("reward_death_value", 10)
        self.reward_scale = kwargs.get("reward_scale", True)
        self.reward_scale_rate = kwargs.get("reward_scale_rate", 20)
        self.reward_defeat = kwargs.get("reward_defeat", -100)
        self.reward_only_positive = kwargs.get("reward_only_positive", False)
        self.max_reward = (self.reward_win + 2 * self.reward_death_value * self.blue_agents_num
                           + 200*self.blue_agents_num)
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
            self.action_spaces = [MultiDiscrete([9, 9, 9, 5, 2, 2, 5]) for _ in range(self.n_agents)]
            self.act_feature_list = act_feature_list
            self.act_feature_list.append("change-target")
        else:
            self.action_spaces = [MultiDiscrete([9, 9, 9, 5, 2, 2]) for _ in range(self.n_agents)]
            self.act_feature_list = act_feature_list

        self.action_space = self.action_spaces[0]
        self.action_map = {}
        self.action_map["fcs/aileron-cmd-norm"] = action_aileron
        self.action_map["fcs/elevator-cmd-norm"] = action_elevator
        self.action_map["fcs/rudder-cmd-norm"] = action_rudder
        self.action_map["fcs/throttle-cmd-norm"] = action_throttle

        # 42 obs
        self.observation_spaces = [
            Box(low=np.float32(-np.inf), high=np.float32(np.inf),
                shape=(255, ), dtype=np.float32) for _ in range(self.n_agents)]
        self.observation_space = self.observation_spaces[0]

        self.action_space_dict = {
            agent: space for agent, space in zip(self.agents, self.action_spaces)
        }
        self.observation_space_dict = {
            agent: space for agent, space in zip(self.agents, self.observation_spaces)
        }

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

        # setup attack variables
        # self.locked_time = np.zeros((self.red_agents_num), dtype=np.float32)
        # self.old_oracle = None
        # self.cur_oracle = None
        

    def reset(self, init=False):
        print(f"reset start...")
        self.red_death = np.zeros((self.red_agents_num), dtype=int)
        self.red_death_missile = np.zeros((self.red_agents_num), dtype=int)
        self.blue_death = np.zeros((self.blue_agents_num), dtype=int)
        self.blue_death_missile = np.zeros((self.blue_agents_num), dtype=int)
        self.cum_rewards = 0
        self._episode_steps = 0
        if init:
            print(self.dict_init)
            self.obs_dict = self.env.reset(self.dict_init)
            self.prev_obs_dict = None
        else:
            print(self.dict_reset)
            self.obs_dict = self.env.reset(self.dict_reset)
            self.prev_obs_dict = None
        print(f"env reset success !")
        obs, obs_op = self.get_obs()
        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] =obs, obs_op

        print(f"reset success!")
        return obs_dict

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""

        self.prev_obs_dict = deepcopy(self.obs_dict)

        ego_action, op_action = actions[0], actions[1]

        self.missile_launch(ego_action)
        infos = [{} for i in range(self.num_agents)]
        dones = np.zeros(self.num_agents, dtype=bool)

        # rewards = [np.zeros((1,), dtype=np.float32)] * self.num_agents

        self.process_actions(ego_action, op_action)

        self.obs_dict = self.env.step(self.current_actions)
        self._episode_steps += 1

        if self.single_agent_mode:
            reward = self.reward_single_agent_mode()
        else:
            reward = self.reward_battle()

        game_done = self.judge_done()
        if game_done:
            terminated = True
            for i in range(self.num_agents):
                dones[i] = True
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            bad_transition = True
            for i in range(self.num_agents):
                dones[i] = True
        else:
            for i in range(self.red_agents_num):
                if self.red_death[i]:
                    dones[i] = True

        obs, obs_op = self.get_obs()

        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = obs, obs_op


        if self.blue_all_dead and not self.red_all_dead:
            if self.reward_sparse and not self.single_agent_mode:
                reward = 1
            else:
                reward += self.reward_win
        elif self.red_all_dead and not self.blue_all_dead:
            if self.reward_sparse and not self.single_agent_mode:
                reward = -1
            else:
                reward += self.reward_defeat

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate
        self.cum_rewards += reward
        print(f"********** reward: {reward} ************")
        
        rewards = [[reward]] * self.num_agents

        return obs_dict, rewards, dones, infos

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
            print(f"obs dim: {obs_agent.shape}")
            # print(f"obs: {obs_agent}")
            obs[self.red_ni_mapping[agent_name]] = obs_agent
            # for i, feature in enumerate(self.obs_feature_list):
            #     obs[self.red_ni_mapping[agent_name]][i] = ego_obs_dict[agent_name][feature]

        for agent_name in op_obs_dict.keys():
            if int(op_obs_dict[agent_name]['DeathEvent']) != 99:
                continue
            obs_agent = self.get_obs_agent(agent_name, camp='blue')
            obs_op[self.blue_ni_mapping[agent_name]] = obs_agent
            # for i, feature in enumerate(self.obs_feature_list):
            #     obs_op[self.blue_ni_mapping[agent_name]][i] = op_obs_dict[agent_name][feature]

        return obs, obs_op

    def get_obs_op(self, camp='blue'):
        """
        opponent info including health and vel infos.
        """
        op_infos = np.zeros((self.blue_agents_num, 18), dtype=np.float32)
        agents = self.camp_2_agents[camp]

        for i, agent_name in enumerate(agents):
            if int(self.obs_dict[camp][agent_name]['DeathEvent']) != 99:
                continue
            op_infos[i, 0] = self.obs_dict[camp][agent_name]['LifeCurrent']
            op_infos[i, 1] = self.obs_dict[camp][agent_name]['velocities/u-fps']
            op_infos[i, 2] = self.obs_dict[camp][agent_name]['velocities/v-fps']
            op_infos[i, 3] = self.obs_dict[camp][agent_name]['velocities/w-fps']
            op_infos[i, 4] = self.obs_dict[camp][agent_name]['velocities/v-north-fps']
            op_infos[i, 5] = self.obs_dict[camp][agent_name]['velocities/v-east-fps']
            op_infos[i, 6] = self.obs_dict[camp][agent_name]['velocities/v-down-fps']
            op_infos[i, 7] = self.obs_dict[camp][agent_name]['velocities/p-rad_sec']
            op_infos[i, 8] = self.obs_dict[camp][agent_name]['velocities/q-rad_sec']
            op_infos[i, 9] = self.obs_dict[camp][agent_name]['velocities/r-rad_sec']
            op_infos[i, 10] = self.obs_dict[camp][agent_name]['velocities/h-dot-fps']
            op_infos[i, 11] = self.obs_dict[camp][agent_name]['velocities/ve-fps']
            op_infos[i, 12] = self.obs_dict[camp][agent_name]['velocities/mach']
            op_infos[i, 13] = self.obs_dict[camp][agent_name]['position/h-sl-ft']
            op_infos[i, 14] = self.obs_dict[camp][agent_name]['attitude/pitch-rad']
            op_infos[i, 15] = self.obs_dict[camp][agent_name]['attitude/roll-rad']
            op_infos[i, 16] = self.obs_dict[camp][agent_name]['attitude/psi-deg']
            op_infos[i, 17] = self.obs_dict[camp][agent_name]['aero/beta-deg']

        obs_op = op_infos.flatten()
        return obs_op

    def get_obs_ego(self, camp='red'):

        ego_agents = self.camp_2_agents[camp]
        ego_agent_num = self.camp_2_agents_num[camp]
        ego_infos = np.zeros((ego_agent_num, 23+10+27), dtype=np.float32)

        for i, name in enumerate(ego_agents):
            if int(self.obs_dict[camp][name]['DeathEvent']) != 99:
                continue
            ego_infos[i, 0] = self.obs_dict[camp][name]['LifeCurrent']
            ego_infos[i, 1] = self.obs_dict[camp][name]['position/h-sl-ft']
            ego_infos[i, 2] = self.obs_dict[camp][name]['attitude/pitch-rad']
            ego_infos[i, 3] = self.obs_dict[camp][name]['attitude/roll-rad']
            ego_infos[i, 4] = self.obs_dict[camp][name]['attitude/psi-deg']
            ego_infos[i, 5] = self.obs_dict[camp][name]['aero/beta-deg']
            ego_infos[i, 6] = self.obs_dict[camp][name]['velocities/u-fps']
            ego_infos[i, 7] = self.obs_dict[camp][name]['velocities/v-fps']
            ego_infos[i, 8] = self.obs_dict[camp][name]['velocities/w-fps']
            ego_infos[i, 9] = self.obs_dict[camp][name]['velocities/v-north-fps']
            ego_infos[i, 10] = self.obs_dict[camp][name]['velocities/v-east-fps']
            ego_infos[i, 11] = self.obs_dict[camp][name]['velocities/v-down-fps']
            ego_infos[i, 12] = self.obs_dict[camp][name]['velocities/p-rad_sec']
            ego_infos[i, 13] = self.obs_dict[camp][name]['velocities/q-rad_sec']
            ego_infos[i, 14] = self.obs_dict[camp][name]['velocities/ve-fps']
            ego_infos[i, 15] = self.obs_dict[camp][name]['velocities/u-fps']
            ego_infos[i, 16] = self.obs_dict[camp][name]['velocities/h-dot-fps']
            ego_infos[i, 17] = self.obs_dict[camp][name]['velocities/mach']
            ego_infos[i, 18] = self.obs_dict[camp][name]['forces/load-factor']
            ego_infos[i, 19] = self.obs_dict[camp][name]['IsOutOfValidBattleArea']
            ego_infos[i, 20] = self.obs_dict[camp][name]['OutOfValidBattleAreaCurrentDuration']
            ego_infos[i, 21] = self.obs_dict[camp][name]['SRAAMCurrentNum']
            ego_infos[i, 22] = self.obs_dict[camp][name]['AMRAAMCurrentNum']
            ego_infos[i, 23+int(self.obs_dict[camp][name]['AimMode'])] = 1
            ego_infos[i, 25+int(self.obs_dict[camp][name]['SRAAM1_CanReload'])] = 1
            ego_infos[i, 27+int(self.obs_dict[camp][name]['SRAAM2_CanReload'])] = 1
            ego_infos[i, 29+int(self.obs_dict[camp][name]['AMRAAMCanReload'])] = 1
            ego_infos[i, 31+int(self.obs_dict[camp][name]['IfPresenceHitting'])] = 1

            if '99' not in str(self.obs_dict[camp][name]['TargetIntoView']):
                indices = self._view_sin(self.obs_dict[camp][name]['TargetIntoView'], base=33)
                ego_infos[i, indices] = 1

            if '99' not in str(self.obs_dict[camp][name]['AllyIntoView']):
                indices = self._view_sin(self.obs_dict[camp][name]['AllyIntoView'], base=38)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['TargetEnterAttackRange']):
                indices = self._view_sin(self.obs_dict[camp][name]['TargetEnterAttackRange'], base=43)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['SRAAMTargetLocked']):
                indices = self._view_sin(self.obs_dict[camp][name]['SRAAMTargetLocked'], base=48)
                ego_infos[i, indices] = 1
            if '99' not in str(self.obs_dict[camp][name]['AMRAAMlockedTarget']):
                indices = self._view_sin(self.obs_dict[camp][name]['AMRAAMlockedTarget'], base=53)
                ego_infos[i, indices] = 1
            ego_infos[i, 58+int(self.obs_dict[camp][name]['MissileAlert'])] = 1

        return ego_infos.flatten()


    def get_obs_agent(self, agent_name, camp='red'):

        
        ctrl_state_feats = np.zeros((5), dtype=np.float32)
        ctrl_cmd_feats = np.zeros((4), dtype=np.float32)
        
        # ctrl state info
        ctrl_state_feats[0] = self.obs_dict[camp][agent_name]['fcs/left-aileron-pos-norm']
        ctrl_state_feats[1] = self.obs_dict[camp][agent_name]['fcs/right-aileron-pos-norm']
        ctrl_state_feats[2] = self.obs_dict[camp][agent_name]['fcs/elevator-pos-norm']
        ctrl_state_feats[3] = self.obs_dict[camp][agent_name]['fcs/rudder-pos-norm']
        ctrl_state_feats[4] = self.obs_dict[camp][agent_name]['fcs/throttle-pos-norm']
        # ctrl cmd info
        ctrl_cmd_feats[0] = self.obs_dict[camp][agent_name]['fcs/aileron-cmd-norm']
        ctrl_cmd_feats[1] = self.obs_dict[camp][agent_name]['fcs/elevator-cmd-norm']
        ctrl_cmd_feats[2] = self.obs_dict[camp][agent_name]['fcs/rudder-cmd-norm']
        ctrl_cmd_feats[3] = self.obs_dict[camp][agent_name]['fcs/throttle-cmd-norm']

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

        # if camp == 'red' and self.obs_dict[camp][agent_name]['MissileAlert']:
        #     self.locked_time[self.red_ni_mapping[agent_name]] += 1
        # else:
        #     self.locked_time[self.blue_ni_mapping[agent_name]] = 0

        # threat info
        threat_info = self._oracal_guiding_feature(agent_name, camp, op_camp)

        if self.apply_agent_ids:
            if camp == 'red':
                agent_id_feats[self.red_ni_mapping[agent_name]] = 1
            else:
                agent_id_feats[self.blue_ni_mapping[agent_name]] = 1

        obs_all = np.concatenate([
            ego_infos.flatten(), ctrl_state_feats.flatten(), 
            ctrl_cmd_feats.flatten(), op_infos.flatten(),
            threat_info.flatten(), agent_id_feats.flatten()
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

        attribute = str(attribute)
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
            print(f"Red all Dead !!!")
        if np.sum(self.blue_death) == self.blue_agents_num:
            self.blue_all_dead = True
            print(f"Blue all Dead !!!")

        return (self.red_all_dead or self.blue_all_dead)

    def seed(self, seed=1):
        random.seed(seed)
        np.random.seed(seed=seed)

    def action_space_sample(self, agent_ids: list = None):
        if agent_ids is None:
            agent_ids = list(range(len(self.agents)))
        actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}

        return actions

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

        weapon_launch += np.sum(self.invalid_launch) * neg_scale
        locked_adv, be_locked_adv = self.locked_reward()
        locked += np.sum(locked_adv * 2 - be_locked_adv)

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
            reward = delta_enemy + delta_deaths - delta_ally - weapon_launch + locked - height - area

        return reward

    def reward_single_agent_mode(self):
        pass

    def missile_launch(self, action):
        self.invalid_launch = np.zeros((self.red_agents_num), dtype=np.float32)

        for al_id, ego_agent_name in enumerate(self.red_agents):
            if int(self.obs_dict['red'][ego_agent_name]['DeathEvent']) != 99:
                continue
            launch = action[al_id, 4]
            if (self.prev_obs_dict['red'][ego_agent_name]['AimMode'] == 0 \
                and int(self.prev_obs_dict['red'][ego_agent_name]['SRAAMTargetLocked']) == 99) or (
                    self.prev_obs_dict['red'][ego_agent_name]['AimMode'] == 1 \
                        and int(self.prev_obs_dict['red'][ego_agent_name]['AMRAAMlockedTarget']) == 99
                ):
                    if launch:
                        self.invalid_launch[al_id] = 1

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

    # def reset_var(self):
    #     self.locked_time = 0
    #     self.old_oracle = None
    #     self.cur_oracle = None

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
