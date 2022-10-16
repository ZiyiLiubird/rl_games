import numpy as np
import math

if __name__ == '__main__':
    def preprocess_actions(self, ego_actions, camp='red'):
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
                    th_ang, th_dis, pitch_reward = self.one_threat_index(ego_agent_name, op_agent_name)
                    reward += 3 * (1. - th_ang) + 2 * (1. - th_dis)
                    reward += pitch_reward
                    print(f'                  dis_threat : {th_dis}')
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

            pitch_reward = 0
            
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
                    pitch_reward +=  -np.sin(pitch)
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
                th_dis = 1. - (100000. - dis_air)*(1 - 0.01)/(100000. - dis_render)
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
            th_ang, th_dis = 0., 0.
        return th_ang, th_dis, pitch_reward

    def calculation_tool(self, x, y, mode='array_angle'):  # 计算工具
        if mode == 'array_angle':  # 输入两向量， 计算两向量夹角
            return np.arccos(x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))) * 180. / np.pi
        elif mode == 'function_parameters':  # 输入两个点，计算一个经过两点的光滑曲线
            a = (x[0] * x[1] - y[0] * y[1]) / (y[1] - x[1])
            b = x[1] * y[1] * (x[0] - y[0]) / (y[1] - x[1])
            return np.array([a, b])
        elif mode == 'distance_threat':  # 输入距离及参数， 计算距离威胁度
            return y[1] / (y[0] + x)
        else:
            print('Error in calculation_tool !! ')
            return None

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
            # print(self.dict_init)
            self.obs_dict = self.env.reset(self.dict_init)
            self.prev_obs_dict = None
        else:
            # print(self.dict_reset)
            self.obs_dict = self.env.reset(self.dict_reset)
            self.prev_obs_dict = None
        # print(f"env reset success !")
        obs, obs_op = self.get_obs()
        obs_dict = {}
        obs_dict['obs'], obs_dict['obs_op'] = obs, obs_op
        self.red_weapon_state = dict()
        for ego_agent_name in self.camp_2_agents['red']:
            self.red_weapon_state[ego_agent_name] = {}
        for ego_agent_name in self.camp_2_agents['blue']:
            self.red_weapon_state[ego_agent_name] = {}


        return obs_dict