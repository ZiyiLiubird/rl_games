import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
from rl_games.algos_pfsp.pfsp_player_pool import PFSPPlayerPool, SinglePlayer, PFSPPlayerThreadPool, PFSPPlayerProcessPool, \
    PFSPPlayerVectorizedPool
from rl_games.algos_torch import a2c_continuous
from rl_games.learning import amp_agent
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
import torch
from torch import optim
from tensorboardX import SummaryWriter
import torch.distributed as dist


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(args, file)
    file.close()


class PFSPAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.player_pool_type = self.config.get('player_pool_type', '')
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.max_his_player_num = self.config.get('player_pool_length', 3)

        if self.config['op_load_path']:
            self.init_op_model = self.create_model()
            self.restore_op(self.config['op_load_path'])
        else:
            self.init_op_model = self.model
        self.players_dir = os.path.join(self.experiment_dir, 'policy_dir')
        os.makedirs(self.players_dir, exist_ok=True)
        self.update_win_rate = self.config.get('update_win_rate', 0.7)
        self.num_opponent_agents = self.num_agents
        self.player_pool = self._build_player_pool(params)

        self.games_to_check = self.config.get('games_to_check', 100)
        self.now_update_steps = 0
        self.max_update_steps = self.config.get('max_update_steps', 5000)
        self.update_op_num = 0
        self.update_player_pool(self.init_op_model, player_idx=self.update_op_num)
        self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))

        assert self.num_actors % self.max_his_player_num == 0

    def _build_player_pool(self, params):
        if self.player_pool_type == 'vectorized':
            vector_model_config = self.base_model_config
            vector_model_config['num_envs'] = self.num_actors * self.num_opponent_agents
            vector_model_config['population_size'] = self.max_his_player_num

            return PFSPPlayerVectorizedPool(max_length=self.max_his_player_num, device=self.device,
                                            vector_model_config=vector_model_config, params=params)
        else:
            return PFSPPlayerPool(max_length=self.max_his_player_num, device=self.device)

    def play_steps(self):
        self.set_eval()
        update_list = self.update_list
        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict_op = self.get_action_values(self.obs, is_op=True)
                res_dict = self.get_action_values(self.obs)
            # print(f"self.obs shape: {self.obs['obs'].shape}")
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            # for k in update_list:
            #     self.experience_buffer.update_data(k, n, res_dict[k])
            # if self.has_central_value:
            #     self.experience_buffer.update_data('states', n, self.obs['states'])

            if self.player_pool_type == 'multi_thread':
                self.player_pool.thread_pool.shutdown()
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'], res_dict_op['actions'])
            actions_relabel = infos['action_relabel']
            if isinstance(actions_relabel, np.ndarray):
                actions_relabel = torch.FloatTensor(actions_relabel)

            res_dict['actions'] = actions_relabel
            # print(f"res_dict action: {res_dict['actions']}")
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            if type(infos) == dict:
                if "win_rate" in infos.keys():
                    win_rate = infos['win_rate']
                else:
                    win_rate = 0
            elif type(infos[0]) == dict and "win_rate" in infos[0].keys():
                win_rate = infos[0]['win_rate']
            else:
                win_rate = 0

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            self.player_pool.update_player_metric(infos=infos)
            self.resample_op(env_done_indices.flatten())

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        batch_dict['win_rate'] = win_rate

        return batch_dict

    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_len,:,:,:] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                res_dict_op = self.get_action_values(self.obs, is_op=True)

            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'], res_dict_op['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            if len(all_done_indices) > 0:
                for s in self.rnn_states:
                    s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        return batch_dict

    def env_step(self, ego_actions, op_actions):

        ego_actions, op_actions = self.preprocess_actions(ego_actions), self.preprocess_actions(op_actions)
        obs, rewards, dones, infos = self.vec_env.step([ego_actions, op_actions])
        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(
                dones).to(self.ppo_device), infos

    def env_reset(self):
        """
        support n cooperative agents vs m other agents.
        """
        obs = self.vec_env.reset(init=True)
        
        obs = self.obs_to_tensors(obs)
        return obs

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()
        print(f"self.obs shape: {self.obs['obs'].shape}")

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        self._init_train()

        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()
            
            sum_time = train_info['total_time']
            play_time = train_info['play_time']
            update_time = train_info['update_time']
            a_losses = train_info['actor_loss']
            c_losses = train_info['critic_loss']
            entropies = train_info['entropy']
            kls = train_info['kl']
            last_lr = train_info['last_lr'][-1]
            lr_mul = train_info['lr_mul'][-1]

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir,
                                           'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                               mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                self.update_metric()
                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def update_metric(self):
        tot_win_rate = 0
        tot_games_num = 0
        self.now_update_steps += 1
        # self_player process
        for player in self.player_pool.players:
            win_rate = player.win_rate()
            games = player.games_num()
            self.writer.add_scalar(f'rate/win_rate_player_{player.player_idx}', win_rate, self.epoch_num)
            tot_win_rate += win_rate * games
            tot_games_num += games
        win_rate = tot_win_rate / tot_games_num
        if tot_games_num > self.games_to_check:
            self.check_update_opponent(win_rate)
        self.writer.add_scalar('rate/win_rate', win_rate, self.epoch_num)

    def get_action_values(self, obs, is_op=False):
        processed_obs = self._preproc_obs(obs['obs_op'] if is_op else obs['obs'])
        if not is_op:
            self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }
        with torch.no_grad():
            if is_op:
                res_dict = {
                    "actions": torch.zeros((self.num_actors * self.num_opponent_agents, self.actions_num),
                                           device=self.device),
                    "values": torch.zeros((self.num_actors * self.num_opponent_agents, 1), device=self.device)
                }
                self.player_pool.inference(input_dict, res_dict, processed_obs)
            else:
                res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def resample_op(self, resample_indices):
        """
        For each real env, player occupys num_opponents continous unreal envs.
        """
        for player in self.player_pool.players:
            player.remove_envs(resample_indices)

        for env_idx in resample_indices:
            player = self.player_pool.sample_player()
            player.add_envs(torch.arange(start=env_idx*self.num_opponent_agents, 
                                         end=env_idx*self.num_opponent_agents+self.num_opponent_agents,
                                         dtype=torch.long))

        for player in self.player_pool.players:
            player.reset_envs()

    def restore_op(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.init_op_model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.init_op_model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def check_update_opponent(self, win_rate):
        if win_rate > self.update_win_rate or self.now_update_steps > self.max_update_steps:
            print(f'winrate:{win_rate},add opponent to player pool')
            self.update_op_num += 1
            self.now_update_steps = 0
            self.update_player_pool(self.model, player_idx=self.update_op_num)
            self.player_pool.clear_player_metric()
            self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))
            self.save(os.path.join(self.players_dir, f'policy_{self.update_op_num}'))

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def update_player_pool(self, model, player_idx):
        new_model = self.create_model()
        new_model.load_state_dict(copy.deepcopy(model.state_dict()))
        if hasattr(model, 'running_mean_std'):
            new_model.running_mean_std.load_state_dict(copy.deepcopy(model.running_mean_std.state_dict()))
        player = SinglePlayer(player_idx, new_model, self.device, self.num_actors * self.num_opponent_agents)
        self.player_pool.add_player(player)
