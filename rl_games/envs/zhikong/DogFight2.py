from .env_zhikong import Base_env
import copy
import gym
import numpy as np
import math
from util.util import get_policy
from custom_env.expert_policy import expendable_policy, level_flight_policy, run_off_policy, circuity_policy
import random


class DogFight(Base_env):
    def __init__(self, config=None, render=0):
        Base_env.__init__(self, config, render)
        self.locked_time = 0
        self.old_oracle = None
        self.cur_oracle = None
        self.old_delta_height = None
        self.cur_delta_height = None
        self.aim_mode = 0
        self.invalid_launch = False
        self.launch = False
        self.switch = False
        self.cur_locked = False
        self.old_locked = False
        self.cur_be_locked = False
        self.old_be_locked = False
        self.total_policy_name = ['expendable', 'circuity']
        # self.total_policy_name = ['expendable', 'circuity', 'level', 'run_off']
        self.opponent_policy_name = random.choice(self.total_policy_name)
        self.opponent_policy = get_policy(self.opponent_policy_name)

    def reset_var(self):
        self.locked_time = 0
        self.aim_mode = 0
        self.cur_locked = False
        self.old_locked = False
        self.cur_be_locked = False
        self.old_be_locked = False
        self.old_oracle = None
        self.cur_oracle = None
        self.invalid_launch = False
        self.old_delta_height = None
        self.cur_delta_height = None
        self.launch = False
        self.switch = False
        self.opponent_policy_name = random.choice(self.total_policy_name)
        self.opponent_policy = get_policy(self.opponent_policy_name)

    def set_s_a_space(self):
        """
        define observation space and action space, which is MultiAgent
        observation_space: []
        """
        # action_space包括了
        self.observation_space = {'move': gym.spaces.Box(low=-10, high=10, dtype=np.float32, shape=(71,)),
                                  'weapon': gym.spaces.Box(low=-10, high=10, dtype=np.float32, shape=(71,))}
        self.action_space = {'move': gym.spaces.Discrete(27),
                             'weapon': gym.spaces.Discrete(4)}

    #################################################################
    # 用于使用规则来收集数据
    #################################################################

    def postprocess_action(self, action):
        if self.usage == 'collect':
            return self.collect_action(action)
        else:
            return self.train_action(action)

    def collect_action(self, action):
        action_input = dict()
        obs = self.obs_tot
        action_input['red'] = action['red']
        action_input['blue'] = action['blue']
        if not self.is_done['red']:
            self.launch = action_input['red']['red_0']['fcs/weapon-launch']
            self.switch = action_input['red']['red_0']['switch-missile']
            self.invalid_launch = False
            self.aim_mode = obs['red']['red_0']['AimMode']
        return action_input

    #################################################################
    # 用于测试或者训练，网络生成的动作
    #################################################################
    def train_action(self, action):
        # 发射武器部分去掉了子弹，只留下导弹
        # 这里只定义了红方的动作，蓝方的动作尚未定义，仅采用一个固定动作，
        # 这里可以使用李论师兄的平飞控制器，或者一些基本的控制器，先暂时稳定住飞行
        a_weapon = int(action['weapon'])
        switch_missile_pre_action = int(a_weapon // 2)
        a_weapon %= 2
        weapon_launch_pre_action = a_weapon

        a_move = int(action['move'])
        action_h = a_move // 9
        a_move %= 9
        action_v = a_move // 3
        a_move %= 3
        action_psi = a_move
        action_input = dict()

        # action_input['blue'], action_blue = level_flight_policy(
        #     obs=self.obs_tot, is_done=self.is_done,
        #     flag='blue', value_oracle=self.cur_oracle, step_num=self.step_num)
        action_input['blue'], action_blue = self.opponent_policy(
            obs=self.obs_tot, is_done=self.is_done,
            flag='blue', value_oracle=self.cur_oracle, step_num=self.step_num)
        if not self.is_done['red']:
            obs = self.obs_tot
            self.aim_mode = obs['red']['red_0']['AimMode']
            SRAAMTargetLocked = obs['red']['red_0']['SRAAMTargetLocked']
            AMRAAMlockedTarget = obs['red']['red_0']['AMRAAMlockedTarget']
            red_psi, red_v, red_h = \
                obs['red']['red_0']['attitude/psi-deg'], \
                math.sqrt(obs['red']['red_0']['velocities/u-fps'] ** 2 +
                          obs['red']['red_0']['velocities/v-fps'] ** 2 +
                          obs['red']['red_0']['velocities/w-fps'] ** 2), \
                obs['red']['red_0']['position/h-sl-ft']
            if weapon_launch_pre_action == 1:
                self.launch = True
            else:
                self.launch = False
            if switch_missile_pre_action == 1:
                self.switch = True
            else:
                self.switch = False
            if (self.aim_mode == 0 and SRAAMTargetLocked != 99) or \
                (self.aim_mode == 1 and AMRAAMlockedTarget != 99):
                weapon_launch = 1
            else:
                if weapon_launch_pre_action == 1:
                    self.invalid_launch = True
                else:
                    self.invalid_launch = False
                weapon_launch = 0
            action_input['red'] = {
                'red_0': {
                    'mode': 2,
                    "target_altitude_ft": red_h + (action_h - 1) * 10000,
                    "target_velocity": red_v + (action_v - 1) * 100,
                    "target_track_deg": red_psi + (action_psi - 1) * 90,
                    "fcs/weapon-launch": weapon_launch,
                    "switch-missile": switch_missile_pre_action,
                    "change-target": 99,
                }}
        else:
            action_input['red'] = {
                'red_0': {
                    'mode': 2,
                    "fcs/weapon-launch": 0,
                    "switch-missile": 0,
                    "change-target": 99,
                }}
        return action_input

    # def postprocess_action(self, action):
    #     # 发射武器部分去掉了子弹，只留下导弹
    #     # 这里只定义了红方的动作，蓝方的动作尚未定义，仅采用一个固定动作，
    #     # 这里可以使用李论师兄的平飞控制器，或者一些基本的控制器，先暂时稳定住飞行
    #     action_red = action['red']
    #     action_blue = action['blue']
    #     a_weapon = int(action_red['weapon'])
    #     switch_missile_pre_action = int(a_weapon // 2)
    #     a_weapon %= 2
    #     weapon_launch_pre_action = a_weapon
    #
    #     a_move = int(action_red['move'])
    #     action_h = a_move // 9
    #     a_move %= 9
    #     action_v = a_move // 3
    #     a_move %= 3
    #     action_psi = a_move
    #     action_input = dict()
    #     obs = self.obs_tot
    #     if not self.is_done['red']:
    #         self.aim_mode = obs['red']['red_0']['AimMode']
    #         SRAAMTargetLocked = obs['red']['red_0']['SRAAMTargetLocked']
    #         AMRAAMlockedTarget = obs['red']['red_0']['AMRAAMlockedTarget']
    #         red_psi, red_v, red_h = \
    #             obs['red']['red_0']['attitude/psi-deg'], \
    #             math.sqrt(obs['red']['red_0']['velocities/u-fps'] ** 2 +
    #                       obs['red']['red_0']['velocities/v-fps'] ** 2 +
    #                       obs['red']['red_0']['velocities/w-fps'] ** 2), \
    #             obs['red']['red_0']['position/h-sl-ft']
    #         if weapon_launch_pre_action == 1:
    #             self.launch = True
    #         else:
    #             self.launch = False
    #         if switch_missile_pre_action == 1:
    #             self.switch = True
    #         else:
    #             self.switch = False
    #         if (self.aim_mode == 0 and SRAAMTargetLocked != 99) or \
    #                 (self.aim_mode == 1 and AMRAAMlockedTarget != 99):
    #             weapon_launch = 1
    #         else:
    #             if weapon_launch_pre_action == 1:
    #                 self.invalid_launch = True
    #             else:
    #                 self.invalid_launch = False
    #             weapon_launch = 0
    #         action_input['red'] = {
    #             'red_0': {
    #                 'mode': 2,
    #                 "target_altitude_ft": red_h + (action_h - 1) * 10000,
    #                 "target_velocity": red_v + (action_v - 1) * 100,
    #                 "target_track_deg": red_psi + (action_psi - 1) * 90,
    #                 "fcs/weapon-launch": weapon_launch,
    #                 "switch-missile": switch_missile_pre_action,
    #                 "change-target": 99,
    #             }}
    #     else:
    #         action_input['red'] = {
    #             'red_0': {
    #                 "fcs/weapon-launch": 0,
    #                 "switch-missile": 0,
    #                 "change-target": 99,
    #             }}
    #     a_weapon = int(action_blue['weapon'])
    #     switch_missile_pre_action = int(a_weapon // 2)
    #     a_weapon %= 2
    #     weapon_launch_pre_action = a_weapon
    #
    #     a_move = int(action_blue['move'])
    #     action_h = a_move // 9
    #     a_move %= 9
    #     action_v = a_move // 3
    #     a_move %= 3
    #     action_psi = a_move
    #     if not self.is_done['blue']:
    #         blue_psi, blue_v, blue_h = \
    #             obs['blue']['blue_0']['attitude/psi-deg'], \
    #             math.sqrt(obs['blue']['blue_0']['velocities/u-fps'] ** 2 +
    #                       obs['blue']['blue_0']['velocities/v-fps'] ** 2 +
    #                       obs['blue']['blue_0']['velocities/w-fps'] ** 2), \
    #             obs['blue']['blue_0']['position/h-sl-ft']
    #         action_input['blue'] = {
    #             'blue_0': {
    #                 'mode': 2,
    #                 "target_altitude_ft": blue_h + (action_h - 1) * 10000,
    #                 "target_velocity": blue_v + (action_v - 1) * 100,
    #                 "target_track_deg": blue_psi + (action_psi - 1) * 90,
    #                 "fcs/weapon-launch": weapon_launch_pre_action,
    #                 "switch-missile": switch_missile_pre_action,
    #                 "change-target": 99,
    #             }}
    #     else:
    #         action_input['blue'] = {
    #             'blue_0': {
    #                 "fcs/weapon-launch": 0,
    #                 "switch-missile": 0,
    #                 "change-target": 99,
    #             }}
    #
    #     return action_input

    """
    以下用于测试环境动作
    
    def postprocess_action(self, action):
        # 发射武器部分去掉了子弹，只留下导弹
        # 这里只定义了红方的动作，蓝方的动作尚未定义，仅采用一个固定动作，
        # 这里可以使用李论师兄的平飞控制器，或者一些基本的控制器，先暂时稳定住飞行
        obs = self.obs_tot
        action_input = dict()
        red_psi, red_v, red_h = \
            obs['red']['red_0']['attitude/psi-deg'],\
            math.sqrt(obs['red']['red_0']['velocities/u-fps'] ** 2 +
                      obs['red']['red_0']['velocities/v-fps'] ** 2 +
                      obs['red']['red_0']['velocities/w-fps'] ** 2),\
            obs['red']['red_0']['position/h-sl-ft']
        blue_psi, blue_v, blue_h = \
            obs['blue']['blue_0']['attitude/psi-deg'], \
            math.sqrt(obs['blue']['blue_0']['velocities/u-fps'] ** 2 +
                      obs['blue']['blue_0']['velocities/v-fps'] ** 2 +
                      obs['blue']['blue_0']['velocities/w-fps'] ** 2), \
            obs['blue']['blue_0']['position/h-sl-ft']
        print('current_red (psi v h)', (red_psi, red_v, red_h),
              'current_blue (psi v h)', (blue_psi, blue_v, blue_h))
        # print('sub_step_left', 'current_v', 'target_v', 'delta_v',
        #       obs['blue']['blue_0']['sub_step_left'],
        #       blue_v,
        #       obs['blue']['blue_0']['target_velocity'],
        #       obs['blue']['blue_0']['delta_velocity'],
        #       obs['blue']['blue_0']['target_altitude_ft'],
        #       obs['blue']['blue_0']['target_track_deg']
        #       )
        # if self.step_num == 0:
        #     action_input['red'] = {
        #         'red_0': {
        #             'mode': 2,
        #             "target_altitude_ft": red_h+1000,
        #             "target_velocity": red_v,
        #             "target_track_deg": red_psi,
        #             "fcs/weapon-launch": 0,
        #             "switch-missile": 0,
        #             "change-target": 99,
        #         }}
        #     action_input['blue'] = {
        #         'blue_0': {
        #             'mode': 2,
        #             "target_altitude_ft": blue_h,
        #             "target_velocity": blue_v,
        #             "target_track_deg": blue_psi - 30,
        #             "fcs/weapon-launch": 0,
        #             "switch-missile": 0,
        #             "change-target": 99,
        #         }}
        # else:
        #     action_input['red'] = {
        #             'red_0': {
        #                 "fcs/weapon-launch": 0,
        #                 "switch-missile": 0,
        #                 "change-target": 99,
        #             }}
        #     action_input['blue'] = {
        #         'blue_0': {
        #             "fcs/weapon-launch": 0,
        #             "switch-missile": 0,
        #             "change-target": 99,
        #         }}
        action_input['red'] = {
            'red_0': {
                'mode': 2,
                "target_altitude_ft": red_h,
                "target_velocity": red_v,
                "target_track_deg": red_psi,
                "fcs/weapon-launch": 0,
                "switch-missile": 0,
                "change-target": 99,
            }}
        action_input['blue'] = {
            'blue_0': {
                'mode': 2,
                "target_altitude_ft": blue_h,
                "target_velocity": blue_v,
                "target_track_deg": blue_psi - 30,
                "fcs/weapon-launch": 0,
                "switch-missile": 0,
                "change-target": 99,
            }}
        # print('target_red (psi v h)', (red_psi + delta_red_psi, red_v + delta_red_v, red_h + delta_red_h),
        #       'target_blue (psi v h)', (blue_psi + delta_blue_psi, blue_v + delta_blue_v, blue_h + delta_blue_h))
        return action_input
    """

    def postprocess_obs(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can normalize obs or postprocess obs here

        Returns:
            postprocessed obs
        """
        post_process_obs = []
        if not self.is_done['red']:
            post_process_obs.extend(self._alley_state_process())
        else:
            post_process_obs.extend([i for i in range(41)])
        if not self.is_done['blue']:
            post_process_obs.extend(self._enemy_state_process())
        else:
            post_process_obs.extend([i for i in range(23)])
        if (not self.is_done['blue']) and (not self.is_done['red']):
            post_process_obs.extend(self._oracal_guiding_feature())
        else:
            post_process_obs.extend([i for i in range(5)])
        post_process_obs.extend(self.death_event())
        obs_out = np.array(post_process_obs).clip(-10, 10)
        return {'move': obs_out,
                'weapon': obs_out}

    def get_reward(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can calculate reward according to the obs

        Returns:
            calculated reward
        """
        ############################################################
        # 首先考虑幕回报，当这一幕结束时给予的回报
        ############################################################
        if self.is_done['__all__']:
            red_death_event = self.obs_tot['red']['red_0']['DeathEvent']
            blue_death_event = self.obs_tot['blue']['blue_0']['DeathEvent']
            # 以下是专门为防御策略写的回报
            # 1. 两个都没死，战平不给予回报
            if red_death_event == 99 and blue_death_event == 99:
                return {'move': 10, 'weapon': 0}
            # 2. 红蓝同归于尽，武器发射给予奖励，躲避的机动动作给予惩罚
            if red_death_event != 99 and blue_death_event != 99:
                return {'move': -10, 'weapon': 10}
            # 3. 红方死了，蓝方没死，证明机动和武器发射选择都不好，给予惩罚
            if red_death_event != 99:
                return {'move': -10, 'weapon': -10}
            # 4. 红方没死，蓝方死了，证明机动和武器发射选择不错，给予奖励
            else:
                # 4.1 蓝方坠毁，红方的机动动作收到些许奖励，武器收到极少的回报
                if blue_death_event == 0:
                    return {'move': 5, 'weapon': 5}
                # 4.2 蓝方被红方击中，要奖励武器发射及时，并奖励机动动作
                else:
                    return {'move': 0, 'weapon': 10}
        elif self.is_done['red'] or self.is_done['blue']:
            return {'move': 0, 'weapon': 0}
        ############################################################
        # 2. 随后考虑稠密回报，仅使用事件触发的回报
        ############################################################
        if self.mode == 'eval':
            return {'move': 0, 'weapon': 0}
        # # 计算基于事件触发的回报
        weapon_reward = self.weapon_reward()
        locked_advantage, be_locked_advantage = self.lock_reward()
        ###########################################################
        # 威胁指数
        ###########################################################
        advantage_pitch_cur = math.cos(self.cur_oracle[1] * 3.14) - \
                              math.cos(self.cur_oracle[3] * 3.14)
        advantage_azimuth_cur = math.cos(self.cur_oracle[2] * 3.14) - \
                                math.cos(self.cur_oracle[4] * 3.14)
        advantage_pitch_old = math.cos(self.old_oracle[1] * 3.14) - \
                              math.cos(self.old_oracle[3] * 3.14)
        advantage_azimuth_old = math.cos(self.old_oracle[2] * 3.14) - \
                                math.cos(self.old_oracle[4] * 3.14)
        d_height = self.cur_delta_height - self.old_delta_height
        d_advantage_azimuth = advantage_azimuth_cur - advantage_azimuth_old
        d_advantage_pitch = advantage_pitch_cur - advantage_azimuth_cur
        # reward_threat = locked_advantage + be_locked_advantage
        if locked_advantage or be_locked_advantage:
            reward_threat = locked_advantage + be_locked_advantage
        else:
            reward_threat = advantage_pitch_cur / 50 + advantage_azimuth_cur / 50
            reward_threat = np.clip(d_advantage_azimuth, -0.02, 0.02) + \
                            np.clip(d_advantage_pitch, -0.02, 0.02) + \
                            np.clip(d_height, -0.02, 0.02)
        # return {'move': 10 * d + be_locked_advantage, 'weapon': weapon_reward}
        return {'move': reward_threat, 'weapon': weapon_reward}

    def judge_done(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can judge whether is_done according to the obs

        Returns:
            is_done or not
        """
        # try:
        #     life_red = self.obs_tot['red']['red_0']['LifeCurrent']
        # except:
        #     life_red = 0
        # try:
        #     life_blue = self.obs_tot['blue']['blue_0']['LifeCurrent']
        # except:
        #     life_blue = 0
        done = {}
        life_red = self.obs_tot['red']['red_0']['LifeCurrent']
        life_blue = self.obs_tot['blue']['blue_0']['LifeCurrent']
        IfPresenceHitting = self.obs_tot['blue']['blue_0']['IfPresenceHitting']
        # print(self.step_num, life_red, life_blue, IfPresenceHitting)
        if life_red == 0:
            done['red'] = True
        else:
            if life_blue == 0 and IfPresenceHitting == 0:
                done['red'] = True
            else:
                done['red'] = False
        if life_blue == 0:
            done['blue'] = True
        else:
            if life_red == 0 and IfPresenceHitting == 0:
                done['blue'] = True
            else:
                done['blue'] = False
        # if self.obs_tot['red']['red_0']['LifeCurrent'] == 0 or self.obs_tot['blue']['blue_0']['LifeCurrent'] == 0:
        if done['blue'] and done['red']:
            done['__all__'] = True
        else:
            done['__all__'] = False
        # print('self.step_num', self.step_num)
        if self.step_num >= 500:
            done['__all__'] = True
        return done

    def weapon_reward(self):
        weapon_reward = 0.0
        if self.invalid_launch:
            weapon_reward -= 0.05
        if self.launch:
            weapon_reward -= 0.05
        if self.switch:
            weapon_reward -= 0.03
        return weapon_reward

    def lock_reward(self):
        ################################################################
        # 首先判断锁定对手后给予的回报
        ################################################################
        obs = self.obs_tot
        if obs['red']['red_0']['SRAAMTargetLocked'] != 99 or \
                obs['red']['red_0']['AMRAAMlockedTarget'] != 99:
            self.cur_locked = True
            if self.old_locked is False:
                # 如果是刚刚锁定，那给予较大的正向回报
                locked_advantage = 0.05
            else:
                # 如果已经锁定了，那给予持续的正向回报
                locked_advantage = 0.025
        else:
            self.cur_locked = False
            if self.old_locked is True:
                # 如果是脱离锁定了，那么给予负向回报
                locked_advantage = -0.05
            else:
                # 如果一直没有锁定，那么就没有回报
                locked_advantage = 0
        ################################################################
        # 其次判断被导弹锁定后给予的回报
        ################################################################
        if obs['blue']['blue_0']['SRAAMTargetLocked'] != 99 or \
                obs['blue']['blue_0']['AMRAAMlockedTarget'] != 99:
            self.cur_be_locked = True
            if self.old_be_locked is False:
                # 如果是刚刚被锁定，那么就给予更大的负向回报
                be_locked_advantage = -0.05
            else:
                # 如果已经被锁定了，那么持续给予负向回报
                be_locked_advantage = -0.025
        else:
            self.cur_be_locked = False
            if self.old_be_locked is True:
                # 如果是刚刚脱离锁定，那么就给予正向回报
                be_locked_advantage = 0.05
            else:
                # 如果一直没有被锁定，那就正常不给予回报
                be_locked_advantage = 0
        self.old_be_locked = self.cur_be_locked
        self.old_locked = self.cur_locked
        return locked_advantage, be_locked_advantage

    def threat_action(self):
        if not self.is_done['blue']:
            obs_blue = self.obs_tot['blue']['blue_0']
            if obs_blue['AimMode'] == 0:
                if obs_blue['SRAAMTargetLocked'] != 99:
                    weapon_launch = 1
                else:
                    weapon_launch = 0
                if obs_blue['SRAAMCurrentNum'] == 0:
                    switch_missile = 1
                else:
                    switch_missile = 0
            else:
                if obs_blue['AMRAAMlockedTarget'] != 99:
                    weapon_launch = 1
                else:
                    weapon_launch = 0
                if obs_blue['AMRAAMCurrentNum'] == 0:
                    switch_missile = 1
                else:
                    switch_missile = 0
            if not self.step_num:
                blue_action = \
                    {'blue_0': {
                        # "fcs/aileron-cmd-norm": 0,
                        # "fcs/elevator-cmd-norm": 0,
                        # "fcs/rudder-cmd-norm": 0,
                        # "fcs/throttle-cmd-norm": 0,
                        'mode': 3,
                        "simulation/do_simple_trim": 1,
                        "fcs/weapon-launch": 0,
                        "switch-missile": 0,
                        "change-target": 99,
                    }}
            else:
                blue_action = \
                    {'blue_0': {
                        # "fcs/aileron-cmd-norm": a_move[0],
                        # "fcs/elevator-cmd-norm": a_move[1],
                        # "fcs/rudder-cmd-norm": a_move[2],
                        # "fcs/throttle-cmd-norm": a_move[3],
                        # "simulation/do_simple_trim": 1,
                        'mode': 3,
                        # "simulation/do_simple_trim": 1,
                        "fcs/weapon-launch": weapon_launch,
                        "switch-missile": switch_missile,
                        "change-target": 99,
                    }}
        else:
            blue_action = \
                {'blue_0': {
                    # "simulation/do_simple_trim": 1,
                    'mode': 3,
                    "fcs/weapon-launch": 0,
                    "switch-missile": 0,
                    "change-target": 99,
                }}
        return blue_action

    def _alley_state_process(self):
        """
        :return: 返回友方与自己的状态信息，因为是友方，所以所有的信息都是可观的
        """
        post_process_obs = []
        # 自身状态基本的信息: 生命值
        post_process_obs.extend(self._base_state_process(flag='red'))
        # 自身状态信息，包括高度，俯仰姿态角
        post_process_obs.extend(self._body_state_process(flag='red'))
        # 控制变量的信息，包括机翼所在的位置
        post_process_obs.extend(self._control_state_process(flag='red'))
        # 速度与角速度的信息，飞机自身的速度，负载系数，出界时间
        post_process_obs.extend(self._v_w_process(flag='red'))
        # 武器的信息，子弹数量，导弹数量等
        post_process_obs.extend(self._weapon_process(flag='red'))
        # 进入视野范围：盟友进入视野，对手进入视野，对手进入攻击区域，导弹锁定信息，导弹锁定时长
        post_process_obs.extend(self._view_process(flag='red'))
        # print('_alley_state_process', len(post_process_obs))
        return post_process_obs

    def _enemy_state_process(self):
        """
        :return: 返回敌方的状态信息，部分信息不可观
        """
        post_process_obs = []
        # 自身状态基本的信息: 生命值，可以大致知道对手的血量
        post_process_obs.extend(self._base_state_process(flag='blue'))
        # 自身状态信息，包括高度，俯仰姿态角，我方无法得到对手准确的姿态角信息，因此此处不启用
        # post_process_obs.extend(self._body_state_process(flag='blue'))
        # 控制变量的信息，我方无法获得对手飞机的控制变量状态，因此此处不启用
        # post_process_obs.extend(self._control_state_process(flag='blue'))
        # 速度与角速度的信息，可以通过机载雷达获得对手的速度信息
        post_process_obs.extend(self._v_w_process(flag='blue'))
        # 武器的信息，TODO 此处按道理来说可以推测出对手的武器剩余信息，但是好想不能直接获得，这里后续仍需要修改
        post_process_obs.extend(self._weapon_process(flag='blue'))
        # 进入视野范围信息，盟友进入视野，对手进入视野，对手进入攻击区域，导弹锁定信息，导弹锁定时长，此处不启用
        # post_process_obs.extend(self._view_process(flag='blue'))
        # print('enemy', len(post_process_obs))
        return post_process_obs

    def _oracal_guiding_feature(self):
        """
        计算红方对蓝方的威胁指数以及蓝方对红方的威胁指数
        没有显示的计算威胁指数，而是获得了红方相对蓝方以及蓝方相对红方的距离，和两者攻击angle的差值
        :return: 红蓝相对距离与角度，全是相对关系
        """
        obs_red = self.obs_tot['red']['red_0']
        obs_blue = self.obs_tot['blue']['blue_0']
        ###########################################################
        # 以下的坐标都是在ECEF(Earth-Centered, Earth-Fixed)坐标系下的，这个坐标系原点在地球球心，随着地球的转动而转动
        ###########################################################
        x_red, y_red, z_red = self.latitude_2_xyz(obs_red['position/long-gc-deg'],
                                                  obs_red['position/lat-geod-deg'],
                                                  obs_red['position/h-sl-ft'] * 0.3048)
        x_blue, y_blue, z_blue = self.latitude_2_xyz(obs_blue['position/long-gc-deg'],
                                                     obs_blue['position/lat-geod-deg'],
                                                     obs_blue['position/h-sl-ft'] * 0.3048
                                                     )
        """
        # 本部分是根据ECEF坐标系计算的，可能有误，暂时放在这里来以防万一
        red2blue = (x_blue - x_red, y_blue - y_red, z_blue - z_red)
        blue2red = (x_red - x_blue, y_red - y_blue, z_red - z_blue)
        ############################################################
        # 红方变量
        ############################################################
        pitch = obs_red['attitude/pitch-rad']
        psi = obs_red['attitude/psi-deg'] / 180 * 3.14
        # 根据俯仰角和航向角来计算当前的机身朝向，用以近似为速度的方向
        velocity_red = (math.cos(pitch) * math.cos(psi), math.cos(pitch) * math.sin(psi), math.sin(pitch))
        ############################################################
        # 蓝方变量
        ############################################################
        pitch = obs_blue['attitude/pitch-rad']
        psi = obs_blue['attitude/psi-deg'] / 180 * 3.14
        # 根据俯仰角和航向角来计算当前的机身朝向，用以近似为速度的方向
        velocity_blue = (math.cos(pitch) * math.cos(psi), math.cos(pitch) * math.sin(psi), math.sin(pitch))
        # 获取红方速度与红蓝连线的夹角，夹角越小证明攻击可能性越高
        theta_red = self.vec2angle(red2blue, velocity_red)
        theta_blue = self.vec2angle(blue2red, velocity_blue)
        """
        # 1. 首先计算红方与蓝方之间的距离，缩小其尺度，以10公里为标准
        distance = math.sqrt((x_blue - x_red) ** 2 + (y_blue - y_red) ** 2 + (z_blue - z_red) ** 2) / 10000
        # 2. 红蓝连线在北天东坐标系下的目标俯仰角与目标航向角，与目前的俯仰角与航向角做差可得到
        # 红方视角，该部分首先算出目标俯仰与航向，随后得到当前的真实俯仰与航向，做差即可得到要调整的大小
        elevation, azimuth = self.look_vector((x_red, y_red, z_red), (x_blue, y_blue, z_blue))
        d_elevation_red = (elevation - obs_red['attitude/pitch-rad'] * 180 / 3.14)
        d_azimuth_red = (azimuth - obs_red['attitude/psi-deg'])
        d_elevation_red_nor = self.nor_deg(d_elevation_red)
        d_azimuth_red_nor = self.nor_deg(d_azimuth_red)
        # 蓝方视角，该部分按道理应该根据速度计算，但是为了简化计算过程，将位姿直接近似成了速度的方向
        elevation, azimuth = self.look_vector((x_blue, y_blue, z_blue), (x_red, y_red, z_red))
        d_elevation_blue = (elevation - obs_blue['attitude/pitch-rad'] * 180 / 3.14)
        d_azimuth_blue = (azimuth - obs_blue['attitude/psi-deg'])
        d_elevation_blue_nor = self.nor_deg(d_elevation_blue)
        d_azimuth_blue_nor = self.nor_deg(d_azimuth_blue)
        post_process_obs = [distance, d_elevation_red_nor, d_azimuth_red_nor,
                            d_elevation_blue_nor, d_azimuth_blue_nor]
        # print('oracal_guiding', post_process_obs)
        delta_h = (obs_red['position/h-sl-ft'] - obs_blue['position/h-sl-ft']) / \
                  max(obs_red['position/h-sl-ft'], obs_blue['position/h-sl-ft'])
        self.old_oracle = self.cur_oracle if self.cur_oracle else post_process_obs
        self.cur_oracle = post_process_obs
        self.old_delta_height = self.cur_delta_height if self.cur_delta_height else delta_h
        self.cur_delta_height = delta_h
        # print(self.cur_delta_height, self.old_delta_height, self.cur_delta_height-self.old_delta_height)
        return post_process_obs

    def death_event(self):
        post_process_obs = []
        try:
            red_death = self.obs_tot['red']['red_0']['DeathEvent']
            if red_death == 99:
                post_process_obs.append(2)
            elif red_death == 0:
                post_process_obs.append(0)
            else:
                post_process_obs.append(1)
        except Exception as e:
            post_process_obs.append(0)

        try:
            blue_death = self.obs_tot['blue']['blue_0']['DeathEvent']
            if blue_death == 99:
                post_process_obs.append(2)
            elif blue_death == 0:
                post_process_obs.append(0)
            else:
                post_process_obs.append(1)
        except Exception as e:
            post_process_obs.append(0)
        return post_process_obs

    def nor_deg(self, deg):
        # normalized to [-1, 1]
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg / 180

    def vec2angle(self, vec1, vec2):
        v1 = 0
        v2 = 0
        v1_v2 = 0
        v_lenth = len(vec1)
        for i in range(v_lenth):
            v1 += vec1[i] ** 2
            v2 += vec2[i] ** 2
            v1_v2 += (vec1[i] * vec2[i])
        # cosθ = v1v2 / (|v1||v2|)
        return math.acos(v1_v2/(math.sqrt(v1 * v2) + 1e-6))

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
        return elevation, azimuth

    def _base_state_process(self, flag):
        post_process_obs = []
        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        # 生命值
        post_process_obs.append(obs['LifeCurrent'] / 200)
        return post_process_obs

    def _body_state_process(self, flag):
        post_process_obs = []
        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        # 高度
        post_process_obs.append(obs['position/h-sl-ft'] / 10000)
        # 俯仰 翻滚 航向 侧滑角等 按道理来说应该以这个角度为参考的初始坐标系，来计算对手在这个坐标系下的角度信息
        post_process_obs.append(obs['attitude/pitch-rad'] / 3.14)
        post_process_obs.append(obs['attitude/roll-rad'] / 3.14)
        post_process_obs.append(obs['attitude/psi-deg'] / 180)
        post_process_obs.append(obs['aero/beta-deg'] / 180)

        # 经纬度，这个是绝对坐标，已经从状态中剔除，采用了相对状态
        # post_process_obs.append(obs['position/lat-geod-deg'])
        # post_process_obs.append(obs['position/long-gc-deg'])

        return post_process_obs

    def _control_state_process(self, flag):
        post_process_obs = []
        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        # 机体被控制的状态
        post_process_obs.append(obs['fcs/left-aileron-pos-norm'])
        post_process_obs.append(obs['fcs/right-aileron-pos-norm'])
        post_process_obs.append(obs['fcs/elevator-pos-norm'])
        post_process_obs.append(obs['fcs/rudder-pos-norm'])
        post_process_obs.append(obs['fcs/throttle-pos-norm'])
        post_process_obs.append(obs['gear/gear-pos-norm'])
        return post_process_obs

    def _v_w_process(self, flag):
        post_process_obs = []

        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        # xyz速度 ft/s
        post_process_obs.append(obs['velocities/u-fps'] / 1000)
        post_process_obs.append(obs['velocities/v-fps'] / 1000)
        post_process_obs.append(obs['velocities/w-fps'] / 1000)
        # 北方向，东方向，下方向速度 ft/s
        post_process_obs.append(obs['velocities/v-north-fps'] / 1000)
        post_process_obs.append(obs['velocities/v-east-fps'] / 1000)
        post_process_obs.append(obs['velocities/v-down-fps'] / 1000)
        # 翻转 俯仰 偏航等角速度速率 rad/s
        post_process_obs.append(obs['velocities/p-rad_sec'])
        post_process_obs.append(obs['velocities/q-rad_sec'])
        post_process_obs.append(obs['velocities/r-rad_sec'])
        # 速率
        post_process_obs.append(obs['velocities/h-dot-fps'] / 1000)
        post_process_obs.append(obs['velocities/mach'])
        # 负载系数
        post_process_obs.append(obs['forces/load-factor'])
        # 是否出界以及出界的时间
        post_process_obs.append(obs['IsOutOfValidBattleArea'])
        post_process_obs.append(obs['OutOfValidBattleAreaCurrentDuration'])
        return post_process_obs

    def _weapon_process(self, flag):
        # 子弹信息
        post_process_obs = []
        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        post_process_obs.append(obs['BulletCurrentNum'] / 100)
        post_process_obs.append(obs['IfOverHeat'])
        # 导弹信息
        post_process_obs.append(obs['AimMode'])
        post_process_obs.append(obs['SRAAMCurrentNum'])
        post_process_obs.append(obs['SRAAM1_CanReload'])
        post_process_obs.append(obs['SRAAM2_CanReload'])
        post_process_obs.append(obs['AMRAAMCurrentNum'])
        post_process_obs.append(obs['AMRAAMCanReload'])
        return post_process_obs

    def _view_process(self, flag):
        post_process_obs = []
        obs = self.obs_tot['red']['red_0'] if flag == 'red' else self.obs_tot['blue']['blue_0']
        flag_in = 'red' if flag == 'blue' else 'blue'
        # 进入视野信息
        post_process_obs.extend(self._view_sin(obs['AllyIntoView'], flag=flag))
        post_process_obs.extend(self._view_sin(obs['TargetIntoView'], flag=flag_in))
        post_process_obs.extend(self._view_sin(obs['TargetEnterAttackRange'], flag=flag_in))
        # 导弹锁定信息
        post_process_obs.extend(self._view_sin(obs['SRAAMTargetLocked'], flag=flag_in))
        post_process_obs.extend(self._view_sin(obs['AMRAAMlockedTarget'], flag=flag_in))
        # 是否被锁定以及锁定的时长
        post_process_obs.append(obs['MissileAlert'])
        if obs['MissileAlert']:
            self.locked_time += 1
        else:
            self.locked_time = 0
        post_process_obs.append(self.locked_time)
        return post_process_obs

    def _view_sin(self, attribute, flag):
        """
        :param attribute: 要处理的属性值，需要是字符串类型的
        :param flag:
        :return:
        """
        attribute = str(attribute)
        if flag == 'red':
            num = self.red_num
        else:
            num = self.blue_num
        if '99' == attribute[0:2]:
            return [0 for i in range(num)]
        else:
            attribute_list = []
            for i in range(num):
                if str(i) in attribute:
                    attribute_list.append(1)
                else:
                    attribute_list.append(0)
            return attribute_list

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


if __name__ == "__main__":
    env = DogFight(render=False)
    while True:
        obs_tot = env.reset()
        is_done = False
        while not is_done:
            action_input_example = {
                'red': {'red_0': {"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0,
                                  "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,
                                  "fcs/weapon-launch": 0, "change-target": 99,
                                  "switch-missile": 0
                                  }},
                'blue': {'blue_0': {"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0,
                                    "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,
                                    "fcs/weapon-launch": 0, "change-target": 99,
                                    "switch-missile": 0
                                    }}
            }
            obs_tot, reward, is_done, info = env.step(action_input_example)
            # print(obs_tot)
