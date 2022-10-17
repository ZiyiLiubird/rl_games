import os
import os.path as osp
import sys

from wandb import agent

parent_path = osp.dirname(__file__)
# print(parent_path)
sys.path.append(parent_path)

import math
from copy import deepcopy
from collections import OrderedDict, deque, namedtuple
import time
import random
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete, Tuple
from zhikong import comm_interface
from .util import init_info, obs_feature_list, act_feature_list



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
        self.setting = kwargs.get("setting", 0)
        self.enemy_weapon = kwargs.get("enemy_weapon", 0)
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
        self.dict_init = init_info(self.n_agents // 2, reset=False, seed=self.setting)
        self.dict_reset = init_info(self.n_agents//2, seed=self.setting)

        # (aileron)：9
        # (elevator): 9
        # (rudder): 9
        # (throttle): 5
        # (weapon-launch): 2
        # (switch-missile)： 2
        # (change-target): 0/1/12/012/0134. 99 default.

        # if self.change_target:
        #     self.act_feature_list = act_feature_list
        #     self.act_feature_list.append("change-target")
        #     self.action_space = Tuple((Discrete(9), Discrete(9),
        #                             Discrete(9), Discrete(5), Discrete(2), Discrete(2),Discrete(5)))
        # else:
        #     self.act_feature_list = act_feature_list
        #     self.action_space = Tuple((Discrete(9), Discrete(9),
        #                             Discrete(9), Discrete(5), Discrete(2), Discrete(2)))
        self.act_feature_list = act_feature_list
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
        setting = np.random.randint(0, 2)
        self._episode_steps = 0
        if init:
            self.obs_dict = self.env.reset(self.dict_init[setting])
        else:
            self.obs_dict = self.env.reset(self.dict_reset[setting])
        print(f"env reset success !")

        return self.obs_dict

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""

        self.prev_obs_dict = self.obs_dict

        ego_action, op_action = actions[0], actions[1]
        dones = np.zeros(self.num_agents, dtype=bool)
        self.ego_weapon_actions = np.zeros((self.red_agents_num, 2), dtype=np.float32)
        self.op_weapon_actions = np.zeros((self.blue_agents_num, 2), dtype=np.float32)
        self.weapon_actions(self.ego_weapon_actions, camp='red')
        self.weapon_actions(self.op_weapon_actions, camp='blue')
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

    def weapon_actions_(self, ego_actions, camp='red'):
        """
        switch missile and weapon launch
        """
        agents = self.camp_2_agents[camp]
        # srmissile_nums = np.zeros((self.red_agents_num, 1), dtype=np.float32)
        for al_id, agent_name in enumerate(agents):
            if int(self.prev_obs_dict[camp][agent_name]['DeathEvent']) != 99:
                continue
            srmissile_nums = self.prev_obs_dict[camp][agent_name]['SRAAMCurrentNum']
            armissile_nums = self.prev_obs_dict[camp][agent_name]['AMRAAMCurrentNum']
            aim_mode = self.prev_obs_dict[camp][agent_name]['AimMode']
            if srmissile_nums == 0 and armissile_nums != 0 and aim_mode == 0:
                ego_actions[al_id][0] = 1
            elif (armissile_nums == 0 and srmissile_nums != 0) and aim_mode == 1:
                ego_actions[al_id][0] = 1
            elif (armissile_nums != 0 and srmissile_nums !=0) and aim_mode == 0:
                ego_actions[al_id][0] = 1
            else:
                ego_actions[al_id][0] = 0
            sr_locked = self.prev_obs_dict[camp][agent_name]['SRAAMTargetLocked']
            ar_locked = self.prev_obs_dict[camp][agent_name]['AMRAAMlockedTarget']
            print(f"agent name: {agent_name}")
            if agent_name == 'red_0':
                print(f"red_0 sr locked: {sr_locked}")
                print(f"red_0 ar locked: {ar_locked}")
            if int(sr_locked) != 9 or  int(ar_locked) != 9999:
                ego_actions[al_id][1] = 1
            else:
                ego_actions[al_id][1] = 0

        return ego_actions

    def process_actions(self, ego_action, op_action):
        """
        ego_action: (num_agent, action_num)
        op_action: (num_op_agent, action_num)

        """

        self.current_actions['red']['red_0']["fcs/aileron-cmd-norm"] = float(ego_action[0][0])
        self.current_actions['red']['red_0']["fcs/elevator-cmd-norm"] = float(ego_action[0][1])
        self.current_actions['red']['red_0']["fcs/throttle-cmd-norm"] = float(ego_action[0][2])
        self.current_actions['red']['red_0']["switch-missile"] =  float(ego_action[0][3])
        self.current_actions['red']['red_0']["fcs/weapon-launch"] =  float(ego_action[0][4])
        
        if int(self.prev_obs_dict['red']['red_0']['DeathEvent']) == 99:
            self.red_0_lat = self.prev_obs_dict['red']['red_0']['position/lat-geod-deg']
            self.red_0_lon = self.prev_obs_dict['red']['red_0']['position/long-gc-deg']
            self.red_0_vel = self.prev_obs_dict['red']['red_0']['velocities/ve-fps'] * 0.6
            self.red_0_height = self.prev_obs_dict['red']['red_0']['position/h-sl-ft'] * np.random.uniform(0.8, 1.2)

        if int(self.prev_obs_dict['blue']['blue_0']['DeathEvent']) == 99:
            self.blue_0_lat = self.prev_obs_dict['blue']['blue_0']['position/lat-geod-deg']
            self.blue_0_lon = self.prev_obs_dict['blue']['blue_0']['position/long-gc-deg']
            self.blue_0_vel = self.prev_obs_dict['blue']['blue_0']['velocities/ve-fps']
            self.blue_0_height = self.prev_obs_dict['blue']['blue_0']['position/h-sl-ft']
        
        for i in range(int(self.red_agents_num)):
            if i == 0:
                continue
            if int(self.prev_obs_dict['red']['red_'+str(i)]['DeathEvent']) != 99:
                continue

            if self._episode_steps == 0:
                self.current_actions['red']['red_'+str(i)]["mode"] = 1
            self.current_actions['red']['red_'+str(i)]["target_longdeg"] = self.blue_0_lon
            self.current_actions['red']['red_'+str(i)]["target_latdeg"] = self.blue_0_lat
            self.current_actions['red']['red_'+str(i)]["target_velocity"] = self.blue_0_vel
            self.current_actions['red']['red_'+str(i)]["target_altitude_ft"] = self.blue_0_height
            if self.enemy_weapon and self._episode_steps > 7:
                self.current_actions['red']['red_'+str(i)]["switch-missile"]=  float(self.ego_weapon_actions[i][0])
                self.current_actions['red']['red_'+str(i)]["fcs/weapon-launch"] =  float(self.ego_weapon_actions[i][1])

        for i in range(int(self.blue_agents_num)):
            if int(self.prev_obs_dict['blue']['blue_'+str(i)]['DeathEvent']) != 99:
                continue
            if self._episode_steps == 0:
                self.current_actions['blue']['blue_'+str(i)]["mode"] = 1
            self.current_actions['blue']['blue_'+str(i)]["target_longdeg"] = self.red_0_lon
            self.current_actions['blue']['blue_'+str(i)]["target_latdeg"] = self.red_0_lat
            self.current_actions['blue']['blue_'+str(i)]["target_velocity"] = self.red_0_vel
            self.current_actions['blue']['blue_'+str(i)]["target_altitude_ft"] = self.red_0_height
            if self.enemy_weapon and self._episode_steps > 7:
                self.current_actions['blue']['blue_'+str(i)]["switch-missile"] = float(self.op_weapon_actions[i][0])
                self.current_actions['blue']['blue_'+str(i)]["fcs/weapon-launch"] = float(self.op_weapon_actions[i][1])

    def save_obs_and_actions(self):
        actions = np.zeros((1, 5), dtype=np.float32)

        actions[0, 0] = self.obs_dict['red']['red_0']['fcs/aileron-cmd-norm']
        actions[0, 1] = self.obs_dict['red']['red_0']['fcs/elevator-cmd-norm']
        actions[0, 2] = self.obs_dict['red']['red_0']['fcs/throttle-cmd-norm']
        actions[0, 3] = self.obs_dict['red']['red_0']['switch-missile']
        actions[0, 4] = self.obs_dict['red']['red_0']['missile-launch']

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

        return self.red_death[0]

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
