import rl_games.envs.test
from rl_games.common import wrappers
from rl_games.common import tr_helpers
from rl_games.envs.brax import create_brax_env
from rl_games.envs.envpool import create_envpool
from rl_games.envs.cule import create_cule
import gym
from gym.wrappers import FlattenObservation, FilterObservation
import numpy as np
import math


class HCRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.max([-10, reward])


class DMControlWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.observation_space = self.env.observation_space['observations']
        self.observation_space.dtype = np.dtype('float32')

    def reset(self, **kwargs):
        self.num_stops = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info




class DMControlObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def observation(self, obs):
        return obs['observations']


def create_default_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    is_procgen = kwargs.pop('procgen', False)
    limit_steps = kwargs.pop('limit_steps', False)
    env = gym.make(name, **kwargs)

    if frames > 1:
        if is_procgen:
            env = wrappers.ProcgenStack(env, frames, True)
        else:
            env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env

def create_goal_gym_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)

    env = gym.make(name, **kwargs)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    if limit_steps:
        env = wrappers.LimitStepsWrapper(env)
    return env 

def create_connect_four_env(**kwargs):
    from rl_games.envs.connect4_selfplay import ConnectFourSelfPlay
    name = kwargs.pop('name')
    limit_steps = kwargs.pop('limit_steps', False)
    self_play = kwargs.pop('self_play', False)
    if self_play:
        env = ConnectFourSelfPlay(name, **kwargs) 
    else:
        env = gym.make(name, **kwargs)
    return env

def create_atari_gym_env(**kwargs):
    #frames = kwargs.pop('frames', 1)
    name = kwargs.pop('name')
    skip = kwargs.pop('skip',4)
    episode_life = kwargs.pop('episode_life',True)
    wrap_impala = kwargs.pop('wrap_impala', False)
    env = wrappers.make_atari_deepmind(name, skip=skip,episode_life=episode_life, wrap_impala=wrap_impala, **kwargs)
    return env    

def create_dm_control_env(**kwargs):
    frames = kwargs.pop('frames', 1)
    name = 'dm2gym:'+ kwargs.pop('name')
    env = gym.make(name, environment_kwargs=kwargs)
    env = DMControlWrapper(env)
    env = DMControlObsWrapper(env)
    env = wrappers.TimeLimit(env, 1000)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env

def create_sts2(**kwargs):
    from rl_games.envs.sts2_env import STSEnv
    env = STSEnv(**kwargs)
    return env

def create_zhikong(**kwargs):
    from rl_games.envs.zhikong.AirCombat_Env import AirCombatEnv
    env = AirCombatEnv(**kwargs)
    return env

def create_zhikong_con(**kwargs):
    from rl_games.envs.zhikong.AirCombat_ConEnv import AirCombatConEnv
    env = AirCombatConEnv(**kwargs)
    return env

def create_smac(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    frames = kwargs.pop('frames', 1)
    transpose = kwargs.pop('transpose', False)
    flatten = kwargs.pop('flatten', True)
    has_cv = kwargs.get('central_value', False)
    env = SMACEnv(name, **kwargs)
    
    
    if frames > 1:
        if has_cv:
            env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=False, flatten=flatten)
        else:
            env = wrappers.BatchedFrameStack(env, frames, transpose=False, flatten=flatten)
    return env

def create_smac_cnn(name, **kwargs):
    from rl_games.envs.smac_env import SMACEnv
    has_cv = kwargs.get('central_value', False)
    frames = kwargs.pop('frames', 4)
    transpose = kwargs.pop('transpose', False)
    env = SMACEnv(name, **kwargs)
    if has_cv:
        env = wrappers.BatchedFrameStackWithStates(env, frames, transpose=transpose)
    else:
        env = wrappers.BatchedFrameStack(env, frames, transpose=transpose)
        
    return env

def create_test_env(name, **kwargs):
    import rl_games.envs.test
    env = gym.make(name, **kwargs)
    return env

def create_multiwalker_env(**kwargs):
    from rl_games.envs.multiwalker import MultiWalker
    env = MultiWalker('', **kwargs) 

    return env

def create_diambra_env(**kwargs):
    from rl_games.envs.diambra.diambra import DiambraEnv
    env = DiambraEnv(**kwargs)
    return env

def create_env(name, **kwargs):
    steps_limit = kwargs.pop('steps_limit', None)
    env = gym.make(name, **kwargs)
    if steps_limit is not None:
        env = wrappers.TimeLimit(env, steps_limit)
    return env

configurations = {
    'CartPole-v1' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs : gym.make('CartPole-v1'),
    },
    'CartPoleMaskedVelocity-v1' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs : wrappers.MaskVelocityWrapper(gym.make('CartPole-v1'), 'CartPole-v1'),
    },
    'MountainCarContinuous-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda **kwargs  : gym.make('MountainCarContinuous-v0'),
    },
    'MountainCar-v0' : {
        'vecenv_type' : 'RAY',
        'env_creator' : lambda : gym.make('MountainCar-v0'),
    },
    'Acrobot-v1' : {
        'env_creator' : lambda **kwargs  : gym.make('Acrobot-v1'),
        'vecenv_type' : 'RAY'
    },
    'Pendulum-v0' : {
        'env_creator' : lambda **kwargs  : gym.make('Pendulum-v0'),
        'vecenv_type' : 'RAY'
    },
    'LunarLander-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLander-v2'),
        'vecenv_type' : 'RAY'
    },
    'PongNoFrameskip-v4' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_atari_deepmind('PongNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'BreakoutNoFrameskip-v4' : {
        'env_creator' : lambda  **kwargs :  wrappers.make_atari_deepmind('BreakoutNoFrameskip-v4', skip=4,sticky=False),
        'vecenv_type' : 'RAY'
    },
    'MsPacmanNoFrameskip-v4' : {
        'env_creator' : lambda  **kwargs :  wrappers.make_atari_deepmind('MsPacmanNoFrameskip-v4', skip=4),
        'vecenv_type' : 'RAY'
    },
    'CarRacing-v0' : {
        'env_creator' : lambda **kwargs  :  wrappers.make_car_racing('CarRacing-v0', skip=4),
        'vecenv_type' : 'RAY'
    },
    'LunarLanderContinuous-v2' : {
        'env_creator' : lambda **kwargs  : gym.make('LunarLanderContinuous-v2'),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalker-v3' : {
        'env_creator' : lambda **kwargs  : create_env('BipedalWalker-v3', **kwargs),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerCnn-v3' : {
        'env_creator' : lambda **kwargs  : wrappers.FrameStack(HCRewardEnv(gym.make('BipedalWalker-v3')), 4, False),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcore-v3' : {
        'env_creator' : lambda **kwargs  : gym.make('BipedalWalkerHardcore-v3'),
        'vecenv_type' : 'RAY'
    },
    'BipedalWalkerHardcoreCnn-v3' : {
        'env_creator' : lambda : wrappers.FrameStack(gym.make('BipedalWalkerHardcore-v3'), 4, False),
        'vecenv_type' : 'RAY'
    },
    'smac' : {
        'env_creator' : lambda **kwargs : create_smac(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'smac_cnn' : {
        'env_creator' : lambda **kwargs : create_smac_cnn(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'sts2': {
        'env_creator' : lambda **kwargs : create_sts2(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'sts2_sp': {
        'env_creator' : lambda **kwargs : create_sts2(**kwargs),
        'vecenv_type' : 'SPRAY'
    },
    'zhikong': {
        'env_creator' : lambda **kwargs : create_zhikong(**kwargs),
        'vecenv_type' : 'SPRAY'
    },
    'zhikong_con': {
        'env_creator' : lambda **kwargs : create_zhikong_con(**kwargs),
        'vecenv_type' : 'SPRAY'
    },
    'dm_control' : {
        'env_creator' : lambda **kwargs : create_dm_control_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_gym' : {
        'env_creator' : lambda **kwargs : create_default_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'openai_robot_gym' : {
        'env_creator' : lambda **kwargs : create_goal_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'atari_gym' : {
        'env_creator' : lambda **kwargs : create_atari_gym_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'test_env' : {
        'env_creator' : lambda **kwargs : create_test_env(kwargs.pop('name'), **kwargs),
        'vecenv_type' : 'RAY'
    },
    'connect4_env' : {
        'env_creator' : lambda **kwargs : create_connect_four_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'multiwalker_env' : {
        'env_creator' : lambda **kwargs : create_multiwalker_env(**kwargs),
        'vecenv_type' : 'RAY'
    },
    'diambra': {
        'env_creator': lambda **kwargs: create_diambra_env(**kwargs),
        'vecenv_type': 'RAY'
    },
    'brax' : {
        'env_creator': lambda **kwargs: create_brax_env(**kwargs),
        'vecenv_type': 'BRAX' 
    },
    'envpool': {
        'env_creator': lambda **kwargs: create_envpool(**kwargs),
        'vecenv_type': 'ENVPOOL'
    },
    'cule': {
        'env_creator': lambda **kwargs: create_cule(**kwargs),
        'vecenv_type': 'CULE'
    },
}

def get_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    result_shapes['value_size'] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    '''
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    
    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    '''
    if hasattr(env, "value_size"):    
        result_shapes['value_size'] = env.value_size
    print(result_shapes)
    return result_shapes

def get_obs_and_action_spaces_from_config(config):
    env_config = config.get('env_config', {})
    env = configurations[config['env_name']]['env_creator'](**env_config)
    result_shapes = get_env_info(env)
    env.close()
    return result_shapes


def register(name, config):
    configurations[name] = config