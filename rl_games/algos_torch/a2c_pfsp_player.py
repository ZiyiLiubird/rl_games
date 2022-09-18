# License: see [LICENSE, LICENSES/isaacgymenvs/LICENSE]
from mimetypes import init
import os
import time
import torch
import gym
import numpy as np
from rl_games.algos_torch import players
import random
from rl_games.algos_torch import torch_ext
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.common.player import BasePlayer
from .pfsp_player_pool import PFSPPlayerPool, PFSPPlayerVectorizedPool, PFSPPlayerThreadPool, PFSPPlayerProcessPool, \
    SinglePlayer
import matplotlib.pyplot as plt

from multielo import MultiElo


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class SPPlayer(BasePlayer):
    def __init__(self, params):
        params['config']['device_name'] = params['player']['device']
        super().__init__(params)
        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.is_discrete = True
        # self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        # self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]
        self.is_rnn = False
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.policy_timestep = []
        self.policy_op_timestep = []
        self.params = params
        self.num_actors = 1
        self.num_opponents = params['player']['num_agents'] // 2
        self.num_agents = self.num_opponents
        self.max_steps = 1000
        self.update_op_num = 0
        self.players_per_env = []

    def restore(self, load_dir):
        model = self.load_model(load_dir)
        self.ego_player = SinglePlayer(player_idx=0, model=model, device=self.device,
                                       obs_batch_len=self.num_actors * self.num_opponents)
        self.restore_op(self.params['player']['op_load_path'])

    def restore_op(self, load_dir):
        model = self.load_model(load_dir)
        self.op_player = SinglePlayer(player_idx=0, model=model, device=self.device,
                                      obs_batch_len=self.num_actors * self.num_opponents)

    def run(self):
        n_games = self.games_num
        n_game_life = self.n_game_life
        is_determenistic = True
        sum_rewards = 0
        sum_steps = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        print(f'games_num:{n_games}')
        need_init_rnn = self.is_rnn
        for i in range(n_games):
            if games_played >= n_games:
                break
            if i == 0:
                inits = True
            else:
                inits = False
            obses = self.env_reset(self.env, init=inits)
            batch_size = 1
            batch_size = self.get_batch_size(obses['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            for n in range(self.max_steps):
                action = self.get_action(obses['obs'], is_determenistic)
                action_op = self.get_action(obses['obs_op'], is_determenistic, is_op=True)
                obses, r, done, info = self.env_step(self.env, action, action_op)
                # print(r[])
                # r = torch.from_numpy(r[0], dtype=torch.float32, device=self.device)
                # cr += r
                steps += 1
                if np.all(done):
                    break
                # all_done_indices = done.nonzero(as_tuple=False)
                # done_indices = all_done_indices
                # done_count = len(done_indices)
                # games_played += done_count
                # if self.record_elo:
                #     self._update_rating(info, all_done_indices.flatten())
                # if done_count > 0:

                    # cur_rewards = cr[done_indices].sum().item()
                    # cur_steps = steps[done_indices].sum().item()

                    # cr = cr * (1.0 - done.float())
                    # steps = steps * (1.0 - done.float())
                    # sum_rewards += cur_rewards
                    # sum_steps += cur_steps

                    # game_res = 0.0
                    # if isinstance(info, dict):
                    #     if 'battle_won' in info:
                    #         print_game_res = True
                    #         game_res = info.get('battle_won', 0.5)
                    #     if 'scores' in info:
                    #         print_game_res = True
                    #         game_res = info.get('scores', 0.5)
                    # if self.print_stats:
                    #     if print_game_res:
                    #         print('reward:', cur_rewards / done_count,
                    #               'steps:', cur_steps / done_count, 'w:', game_res)
                    #     else:
                    #         print('reward:', cur_rewards / done_count,
                    #               'steps:', cur_steps / done_count)

                    # sum_game_res += game_res
                    # if batch_size // self.num_agents == 1 or games_played >= n_games:
                    #     break

    def get_action(self, obs, is_determenistic=False, is_op=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            # data_len = self.num_actors * self.num_opponents #if is_op else self.num_actors
            # res_dict = {
            #     "actions": torch.zeros((data_len, len(self.actions_num)), device=self.device),
            #     "values": torch.zeros((data_len, 1), device=self.device),
            # }
            # print(f"obs:{input_dict['obs']}")
            if is_op:
               res_dict = self.op_player(input_dict)
            else:
                res_dict = self.ego_player(input_dict)
                # self.player_pool.inference(input_dict, res_dict, obs)
        action = res_dict['actions']
        # self.states = res_dict['rnn_states']
        current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        return current_action

    def env_reset(self, env, init):
        obs = env.reset(init)
        obs = self.obs_to_tensors(obs)
        
        return obs

    def env_step(self, env, ego_actions, op_actions):
        actions = [ego_actions, op_actions]
        obs, rewards, dones, infos = env.step(actions)
        # if hasattr(obs, 'dtype') and obs.dtype == np.float64:
        #     obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_tensors(obs), rewards, dones, infos

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def load_model(self, fn):
        model = self.create_model()
        checkpoint = torch_ext.safe_filesystem_op(torch.load, fn, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        return model

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

