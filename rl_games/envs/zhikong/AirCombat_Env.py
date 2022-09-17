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

        # setup agents
        self.red_agents = ["red_" + str(i) for i in range(int(self.red_agents_num))]
        self.blue_agents = ["blue_" + str(i) for i in range(int(self.blue_agents_num))]
        self.camp_2_agents = {}
        self.camp_2_agents['red'] = self.red_agents
        self.camp_2_agents['blue'] = self.blue_agents
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
                shape=(137, ), dtype=np.float32) for _ in range(self.n_agents)]
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
        self.locked_time = 0
        self.old_oracle = None
        self.cur_oracle = None

    def reset(self, init=False):
        print(f"reset start...")
        self._episode_steps = 0
        if init:
            print(self.dict_init)
            self.obs_dict = self.env.reset(self.dict_init)
        else:
            print(self.dict_reset)
            self.obs_dict = self.env.reset(self.dict_reset)
        print(f"env reset success !")
        obs, obs_op = self.get_obs()
        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = self._preproc_obs(obs, obs_op)

        print(f"reset success!")
        return obs_dict

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""

        ego_action, op_action = actions[0], actions[1]

        terminated = False
        bad_transition = False
        infos = [{} for i in range(self.num_agents)]
        dones = np.zeros(self.num_agents, dtype=bool)

        rewards = [np.zeros((1,), dtype=np.float32)] * self.num_agents

        self.process_actions(ego_action, op_action)

        self.obs_dict = self.env.step(self.current_actions)
        if self.judge_done():
            terminated = True
            for i in range(self.num_agents):
                dones[i] = True
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            bad_transition = True
            for i in range(self.num_agents):
                dones[i] = True

        obs, obs_op = self.get_obs()
        # global_state = self.get_state()

        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = self._preproc_obs(obs, obs_op)
        # obs_dict['states'] = global_state

        # rewards = self.reward_battle(self.obs_dict)
        rewards = np.stack(rewards)
        print(f"step success")

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

    def get_obs(self):

        ego_obs_dict = self.obs_dict['red']
        op_obs_dict = self.obs_dict['blue']

        obs = np.zeros((self.red_agents_num, self.observation_space.shape[0]), dtype=np.float32)
        obs_op = np.zeros((self.blue_agents_num, self.observation_space.shape[0]), dtype=np.float32)

        for agent_name in ego_obs_dict.keys():
            if int(ego_obs_dict[agent_name]['DeathEvent']) != 99:
                continue
            obs_agent = self.get_obs_agent(agent_name, camp='red')
            # print(f"obs dim: {obs_agent.shape}")
            print(f"obs: {obs_agent}")
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



    def get_obs_agent(self, agent_name, camp='red'):

        health_feats = np.zeros((1), dtype=np.float32)
        position_feats = np.zeros((5), dtype=np.float32)
        ctrl_state_feats = np.zeros((5), dtype=np.float32)
        ctrl_cmd_feats = np.zeros((4), dtype=np.float32)
        vel_feats = np.zeros((13), dtype=np.float32)
        area_feats = np.zeros((2), dtype=np.float32)
        weapon_feats = np.zeros((2), dtype=np.float32)

        if camp == 'red':
            agent_id_feats = np.zeros(self.red_agents_num, dtype=np.float32)
        else:
            agent_id_feats = np.zeros(self.blue_agents_num, dtype=np.float32)

        health_feats[0] = self.obs_dict[camp][agent_name]['LifeCurrent']
        # position info
        position_feats[0] = self.obs_dict[camp][agent_name]['position/h-sl-ft']
        position_feats[1] = self.obs_dict[camp][agent_name]['attitude/pitch-rad']
        position_feats[2] = self.obs_dict[camp][agent_name]['attitude/roll-rad']
        position_feats[3] = self.obs_dict[camp][agent_name]['attitude/psi-deg']
        position_feats[4] = self.obs_dict[camp][agent_name]['aero/beta-deg']
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
        # vel info
        vel_feats[0] = self.obs_dict[camp][agent_name]['velocities/u-fps']
        vel_feats[1] = self.obs_dict[camp][agent_name]['velocities/v-fps']
        vel_feats[2] = self.obs_dict[camp][agent_name]['velocities/w-fps']
        vel_feats[3] = self.obs_dict[camp][agent_name]['velocities/v-north-fps']
        vel_feats[4] = self.obs_dict[camp][agent_name]['velocities/v-east-fps']
        vel_feats[5] = self.obs_dict[camp][agent_name]['velocities/v-down-fps']
        vel_feats[6] = self.obs_dict[camp][agent_name]['velocities/p-rad_sec']
        vel_feats[7] = self.obs_dict[camp][agent_name]['velocities/q-rad_sec']
        vel_feats[8] = self.obs_dict[camp][agent_name]['velocities/ve-fps']
        vel_feats[9] = self.obs_dict[camp][agent_name]['velocities/u-fps']
        vel_feats[10] = self.obs_dict[camp][agent_name]['velocities/h-dot-fps']
        vel_feats[11] = self.obs_dict[camp][agent_name]['velocities/mach']
        vel_feats[12] = self.obs_dict[camp][agent_name]['forces/load-factor']
        # area info
        area_feats[0] = self.obs_dict[camp][agent_name]['IsOutOfValidBattleArea']
        area_feats[1] = self.obs_dict[camp][agent_name]['OutOfValidBattleAreaCurrentDuration']
        # weapon info
        weapon_feats[0] = self.obs_dict[camp][agent_name]['SRAAMCurrentNum']
        weapon_feats[1] = self.obs_dict[camp][agent_name]['AMRAAMCurrentNum']
        weapon_one_hot1 = np.zeros((2), dtype=np.float32)
        weapon_one_hot2 = np.zeros((2), dtype=np.float32)
        weapon_one_hot3 = np.zeros((2), dtype=np.float32)
        weapon_one_hot4 = np.zeros((2), dtype=np.float32)
        weapon_one_hot5 = np.zeros((2), dtype=np.float32)
        weapon_one_hot1[int(self.obs_dict[camp][agent_name]['AimMode'])] = 1
        weapon_one_hot2[int(self.obs_dict[camp][agent_name]['SRAAM1_CanReload'])] = 1
        weapon_one_hot3[int(self.obs_dict[camp][agent_name]['SRAAM2_CanReload'])] = 1
        weapon_one_hot4[int(self.obs_dict[camp][agent_name]['AMRAAMCanReload'])] = 1
        weapon_one_hot5[int(self.obs_dict[camp][agent_name]['IfPresenceHitting'])] = 1
        # view info
        view_one_hot_1 = np.zeros((5), dtype=np.float32)
        view_one_hot_2 = np.zeros((5), dtype=np.float32)
        view_one_hot_3 = np.zeros((5), dtype=np.float32)
        view_one_hot_4 = np.zeros((5), dtype=np.float32)
        view_one_hot_5 = np.zeros((5), dtype=np.float32)
        view_one_hot_6 = np.zeros((2), dtype=np.float32)

        if '99' not in str(self.obs_dict[camp][agent_name]['TargetIntoView']):
            indices = self._view_sin(self.obs_dict[camp][agent_name]['TargetIntoView'])
            view_one_hot_1[indices] = 1
        if '99' not in str(self.obs_dict[camp][agent_name]['AllyIntoView']):
            indices = self._view_sin(self.obs_dict[camp][agent_name]['AllyIntoView'])
            view_one_hot_2[indices] = 1
        if '99' not in str(self.obs_dict[camp][agent_name]['TargetEnterAttackRange']):
            indices = self._view_sin(self.obs_dict[camp][agent_name]['TargetEnterAttackRange'])
            view_one_hot_3[indices] = 1
        if '99' not in str(self.obs_dict[camp][agent_name]['SRAAMTargetLocked']):
            indices = self._view_sin(self.obs_dict[camp][agent_name]['SRAAMTargetLocked'])
            view_one_hot_4[indices] = 1
        if '99' not in str(self.obs_dict[camp][agent_name]['AMRAAMlockedTarget']):
            indices = self._view_sin(self.obs_dict[camp][agent_name]['AMRAAMlockedTarget'])
            view_one_hot_5[indices] = 1        
        view_one_hot_6[int(self.obs_dict[camp][agent_name]['MissileAlert'])] = 1

        if camp == 'red' and self.obs_dict[camp][agent_name]['MissileAlert']:
            self.locked_time += 1
        else:
            self.locked_time = 0

        # op info
        if camp == 'red':
            obs_op_info = self.get_obs_op(camp='blue')
            op_camp = 'blue'
        else:
            obs_op_info = self.get_obs_op(camp='red')
            op_camp = 'red'

        # threat info

        threat_info = self._oracal_guiding_feature(agent_name, camp, op_camp)

        if self.apply_agent_ids:
            if camp == 'red':
                agent_id_feats[self.red_ni_mapping[agent_name]] = 1
            else:
                agent_id_feats[self.blue_ni_mapping[agent_name]] = 1

        obs_all = np.concatenate([
            health_feats.flatten(), position_feats.flatten(),
            ctrl_state_feats.flatten(), ctrl_cmd_feats.flatten(),
            vel_feats.flatten(), area_feats.flatten(),
            area_feats.flatten(), weapon_feats.flatten(),
            weapon_one_hot1.flatten(), weapon_one_hot2.flatten(),
            weapon_one_hot3.flatten(), weapon_one_hot4.flatten(),
            weapon_one_hot5.flatten(),view_one_hot_1.flatten(),
            view_one_hot_2.flatten(), view_one_hot_3.flatten(),
            view_one_hot_4.flatten(), view_one_hot_5.flatten(),
            view_one_hot_6.flatten(), obs_op_info.flatten(),
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

    def _view_sin(self, attribute):

        attribute = str(attribute)
        attribute_list = []
        for i in attribute:
            if i == '.':
                continue
            attribute_list.append(int(i))
        return attribute_list

    def get_state(self, agents_dict):
        return None

    def judge_done(self):
        red_all_dead = True
        blue_all_dead = True
        # print(f"************obs_dict inner*******************")
        # print(agents_dict)
        for key in self.obs_dict['red'].keys():
            if int(self.obs_dict['red'][key]['DeathEvent']) == 99:
                print(f"red death event: {self.obs_dict['red'][key]['DeathEvent']}")
                red_all_dead = False
                break
        if red_all_dead:
            print("Red all Dead!")
            return red_all_dead
        for key in self.obs_dict['blue'].keys():
            # print(f"blue: keys {agents_dict['blue'][key].keys()}")
            if int(self.obs_dict['blue'][key]['DeathEvent']) == 99:
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

    def reset_var(self):
        self.locked_time = 0
        self.old_oracle = None
        self.cur_oracle = None

    def render(self):
        pass

    def latitude_2_xyz(self, longitude, latitude, height):
        """
        版权声明：本文为CSDN博主「和光尘樾」的原创文章，遵循CC 4.0 BY - SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https: // blog.csdn.net / qq_42482412 / article / details / 111301860
        :param longitude: 经度
        :param latitude:  纬度
        :param height:    高度
        :return: (x, y, z)
        """
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
        """
        引用： https://github.com/JSBSim-Team/jsbsim/discussions/628中答复代码
        涉及到了ECEF坐标系转化到NED坐标系
        :param from_vector: 红方战机的坐标
        :param to_vector:   蓝方战机的坐标
        :return:            红方战机朝向蓝方战机需要的目标俯仰与偏航，这是在北天东坐标系下的定义
        """
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
