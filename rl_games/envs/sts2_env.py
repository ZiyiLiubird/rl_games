from rl_games.envs.env_wrapper import Env
import gym
import numpy as np
import yaml
from rl_games.torch_runner import Runner
import os


class STSEnv(Env):
    def __init__(self, **kwargs):
        super(STSEnv, self).__init__(kwargs, test=kwargs.get('test', False), N_roles=kwargs.get('N_roles', None))
        self.use_central_value = kwargs.pop('central_value', False)
        self.apply_agent_ids = kwargs.get('apply_agent_ids', False)
        self.concat_infos = True
        self.num_agents = self.N_home - self.N_home_ai
        self.num_actions = self.action_dim
        self.action_space = gym.spaces.Discrete(self.num_actions)
        one_hot_agents = 0
        if self.apply_agent_ids:
            one_hot_agents = self.num_agents
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim+one_hot_agents, ), dtype=np.float32) # or Dict
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ), dtype=np.float32) # or Dict
        self.config_path = kwargs.pop('config_path')
        self.is_deterministic = kwargs.pop('is_deterministic', False)

    def reset(self):
        obs_dict = self.reset0()
        self.sum_rewards = 0
        obs_dict['obs'], obs_dict['obs_op'] = self._preproc_obs(obs_dict['obs'], obs_dict['obs_op'])
        return obs_dict

    def step(self, action):
        if self.self_play:
            ego_action, opponent_action = action[0], action[1]
            actions_int = [ego_action, opponent_action]
        else:
            actions_int = action

        obs_dict, reward_dict, done, info = self.step0(actions_int)
        reward = np.array(reward_dict['reward'][0], dtype=np.float32) if self.self_play else np.array(reward_dict['reward'], dtype=np.float32)
        self.sum_rewards += reward
        # if reward < 0:
        #     reward = reward * self.neg_scale
        obs_dict['obs'], obs_dict['obs_op'] = self._preproc_obs(obs_dict['obs'], obs_dict['obs_op'])
        if done:
            info['battle_won'] = np.sign(self.sum_rewards)
        reward_team = [reward] * self.num_agents
        reward_team = np.stack(reward_team)
        done_team = [done] * self.num_agents
        done_team = np.stack(done_team)

        return obs_dict, reward_team, done_team, info

    def render(self, mode=None):
        self.env.render()

    def get_action_mask(self):
        pass

    def get_number_of_agents(self):
        return self.num_agents
