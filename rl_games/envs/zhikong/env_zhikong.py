import socket
import json
import gym
import os
import time
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent


# class Base_env(gym.Env):
class Base_env(MultiAgentEnv):
    """
    (1) 动作输入指令: 字典格式
        Action_input: Dict
    {
        'red':{
            'red_0':{
                'fcs/aileron-cmd-norm':  Float   副翼指令      [-1, 1]
                'fcs/elevator-cmd-norm': Float   升降舵指令    [-1, 1]
                'fcs/rudder-cmd-norm':   Float   方向舵指令    [-1, 1]
                'fcs/throttle-cmd-norm': Float   油门指令      [ 0, 1]
                'fcs/weapon-launch':     Enum    导弹发射指令
                                        0-不发射 (Don't launch)
                                        1-发射导弹 (Launch missile)
                                        2-发射子弹 (Launch bullet)
                'fcs/switch-missile':    Bool    导弹切换指令
                                        0-不变 (Don't switch)
                                        1-切换 (Switch)
                'fcs/change-target':     Enum     切换目标指令
                                        99-不变 (Don't change)
                                        88-由内部程序控制 (Controlled by procedure)
                                        0/1/12/012/0134-优先锁定目标机编号
            }
        }
        ‘blue':{
            'blue_0':{
                与上述格式相同 The same as above
            }
        }
    }
    (2) 初始化指令(init 与 reset)
        {
        'flag': {
            'init': {
                'render':                   Bool        是否显示可视化界面
                }},
        'red': {
            'red_0': {
                "ic/h-sl-ft":               Float       初始高度 [ft]
                "ic/terrain-elevation-ft":  Float       初始地形高度 [ft]
                "ic/long-gc-deg":           Float       初始经度
                "ic/lat-geod-deg":          Float       初始纬度
                "ic/u-fps":                 Float       机体坐标系x轴速度 [ft/s]
                "ic/v-fps":                 Float       机体坐标系y轴速度 [ft/s]
                "ic/w-fps":                 Float       机体坐标系z轴速度 [ft/s]
                "ic/p-rad_sec":             Float       翻滚速率 [rad/s]
                "ic/q-rad_sec":             Float       俯仰速率 [rad/s]
                "ic/r-rad_sec":             Float       偏航速率 [rad/s]
                "ic/roc-fpm":               Float       初始爬升速率 [ft/min]
                "ic/psi-true-deg":          Float       初始航向 [度]
                }},
        'blue': {
            'blue_0': the same as above}

    """
    init_data_example = {
        'flag': {'init': {'render': 0}},
        'red': {
            'red_0': {
                "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.01,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
                 },
        'blue': {
            'blue_0': {
                "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.01,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
                 }}
    reset_data_example = {
        'flag': {'reset': {}},
        'red': {
            'red_0': {
                "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.01,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
                 },
        'blue': {
            'blue_0': {
                "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.01,
                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
                 }}
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
    IP = '127.0.0.1'
    PORT = 4000
    INITIAL = False
    RENDER = 0

    def __init__(self, config=None, render=0):
        """
        :param config: 从RLlib中传输过来的参数，在这个config里面可以传递希望定制的环境变量，譬如ip，render等
        :param ip:     开启软件的ip地址
        :param port:   开启软件的端口号，注意端口号应该是四位的
        :param render: 是否可视化
        :param excute_path: 主要是根据软件位置的不同
        """
        ip = '127.0.0.1'
        port = 4000
        excute_path = '/home/user/linux/ZK.x86_64'
        red_num = 1
        blue_num = 1
        scenes = 3
        if config is not None:
            keys = config.keys()
            ip = config['ip'] if 'ip' in keys else ip
            red_num = config['red_num'] if 'red_num' in keys else red_num
            blue_num = config['blue_num'] if 'blue_num' in keys else blue_num
            scenes = config['scenes'] if 'scenes' in keys else scenes
            excute_path = config['excute_path'] if 'excute_path' in keys else excute_path
            render = int(config['render']) if 'render' in keys else int(render)
        try:
            port = config.worker_index + 8000
        except:
            port = port
        self.IP = ip
        self.PORT = port
        self.RENDER = int(render)
        self.red_num = red_num
        self.blue_num = blue_num
        self.scenes = scenes
        self.excute_path = excute_path
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data = None  # set for debug
        self.INITIAL = False
        self.excute_cmd = f'{excute_path} ip={self.IP} port={self.PORT} ' \
                          f'PlayMode={self.RENDER} ' \
                          f'RedNum={self.red_num} BlueNum={self.blue_num} ' \
                          f'Scenes={self.scenes}'
        print('Creating Env', self.excute_cmd)
        self.unity = os.popen(self.excute_cmd)
        time.sleep(20)
        self._connect()
        self.set_s_a_space()
        self.obs_tot = None
        self.is_done = False
        self.step_num = 0
        print('Env Created')

    def _send_condition(self, data):
        self.socket.send(bytes(data.encode('utf-8')))
        self.data = data

    def _connect(self):
        self.socket.connect((self.IP, self.PORT))

    def _accept_from_socket(self):
        try:
            msg_receive = json.loads(str(self.socket.recv(8192), encoding='utf-8'))
        except:
            print("fail to recieve message from unity")
            print("the last sent data is {}", self.data)
        return msg_receive

    def get_obs(self):
        ask_info = {'flag': 'obs'}
        data = json.dumps(ask_info)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()
        return msg_receive

    def get_obs_red(self):
        global_msg = self.get_obs()
        red_msg = global_msg['red']
        return red_msg

    def get_obs_blue(self):
        global_msg = self.get_obs()
        blue_msg = global_msg['blue']
        return blue_msg

    def reset(self, red_number: int = 1, blue_number: int = 1,
              reset_attribute: dict = reset_data_example):
        reset_attribute = {
            'red': {
                'red_0': {
                    "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.1 * np.random.random(),
                    "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
                     },
            'blue': {
                'blue_0': {
                    "ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.1 * np.random.random(),
                    "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0,
                    "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                    "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
                     }}
        init_info = {'red': reset_attribute['red'],
                     'blue': reset_attribute['blue']}
        self.reset_var()
        if self.INITIAL is False:
            self.INITIAL = True
            init_info['flag'] = {'init': {'render': self.RENDER}}
        else:
            init_info['flag'] = {'reset': {'render': self.RENDER}}

        data = json.dumps(init_info)
        self.is_done = {'__all__': False}
        self.step_num = 0
        self._send_condition(data)
        self.obs_tot = self._accept_from_socket()
        obs = self.postprocess_obs(self.obs_tot)
        return obs

    def reset_var(self):
        pass

    def step(self, action_attribute):
        action_attribute = self.postprocess_action(action_attribute)
        data = json.dumps(action_attribute)
        self._send_condition(data)
        self.obs_tot = self._accept_from_socket()
        obs = self.postprocess_obs(self.obs_tot)
        self.is_done = self.judge_done(self.obs_tot)
        reward = self.get_reward(self.obs_tot)
        info = {}
        self.step_num += 1
        # print('obs', obs)
        # print('self.is_done', self.is_done)
        return obs, reward, self.is_done, info

    # TODO: Need to be overwritten according to different work
    def postprocess_obs(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can normalize obs or postprocess obs here

        Returns:
            postprocessed obs
            {'red_0': obs_red,
            'blue_0': obs_blue}
        """
        raise NotImplementedError

    # TODO: Need to be overwritten according to different work
    def get_reward(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can calculate reward according to the obs

        Returns:
            calculated reward
            {'red_0': reward1, 'blue_0': reward2}
        """
        raise NotImplementedError

    # TODO: Need to be overwritten according to different work
    def judge_done(self, obs):
        """
        Args:
            obs: dict format, include all message back from socket,
                 you can judge whether is_done according to the obs

        Returns:
            is_done or not
            {
                "red_0": False,    # car_0 is still running
                "blue_0": True,     # car_1 is done
                "__all__": False,  # the env is not done
            }
        """
        raise NotImplementedError

    def postprocess_action(self, action):
        """
        Args:
            action: dict format, you can postprocess action according to different tasks
            {"red_0": [1,2,3,4],
            "blue_0": [2,3,4,5]})
        Returns:
            dict format action
            the same as action_input_example
        """
        raise NotImplementedError

    def set_s_a_space(self):
        """
        define state space and action space
        """
        raise NotImplementedError




