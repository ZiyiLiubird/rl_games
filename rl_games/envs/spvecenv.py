import ray
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.env_configurations import configurations
from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
from rl_games.utils.motion_lib import MotionLib
import numpy as np
import gym
import random
from time import sleep
import torch
from gym import spaces

class SPRayWorker:
    def __init__(self, config_name, config, env_id=888):
        config['worker_index'] =  env_id
        self.env = configurations[config_name]['env_creator'](**config)

    def _obs_to_fp32(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        if dv.dtype == np.float64:
                            v[dk] = dv.astype(np.float32)
                else:
                    if v.dtype == np.float64:
                        obs[k] = v.astype(np.float32)
        else:
            if obs.dtype == np.float64:
                obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)

        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        return next_state, reward, is_done, info

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)

    def render(self):
        self.env.render()

    def reset(self, init=False):
        obs = self.env.reset(init)
        obs = self._obs_to_fp32(obs)
        return obs

    def get_action_mask(self):
        return self.env.get_action_mask()

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        else:
            return 1

    def can_concat_infos(self):
        if hasattr(self.env, 'concat_infos'):
            return self.env.concat_infos
        else:
            return False

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if hasattr(self.env, 'use_central_value'):
            info['use_global_observations'] = self.env.use_central_value
        if hasattr(self.env, 'value_size'):
            info['value_size'] = self.env.value_size
        if hasattr(self.env, 'state_space'):
            info['state_space'] = self.env.state_space
        # if hasattr(self.env, 'amp_observation_space'):
        #     info['amp_observation_space'] = self.env.amp_observation_space
        # else:
        #     info['amp_observation_space'] = observation_space

        return info


class SPRayVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)
        self.remote_worker = ray.remote(SPRayWorker)
        worker_index = kwargs.get("worker_index", 888)
        self.workers = [self.remote_worker.remote(self.config_name, kwargs, i+worker_index) for i in range(self.num_actors)]

        if self.seed is not None:
            seeds = range(self.seed, self.seed + self.num_actors)
            seed_set = []
            for (seed, worker) in zip(seeds, self.workers):	        
                seed_set.append(worker.seed.remote(seed))
            ray.get(seed_set)

        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = ray.get(res)

        res = self.workers[0].get_env_info.remote()
        env_info = ray.get(res)
        res = self.workers[0].can_concat_infos.remote()
        can_concat_infos = ray.get(res)
        self.use_global_obs = env_info['use_global_observations']
        self.concat_infos = can_concat_infos
        self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
        self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
        if self.num_agents == 1:
            self.concat_func = np.concatenate
        else:
            self.concat_func = np.concatenate

    def step(self, actions):
        newobs, newobs_op, newstates, newrewards, newdones, newinfos = [], [], [], [], [], []
        ego_actions, op_actions = actions
        res_obs = []
        if self.num_agents == 1:
            for (ego_action, op_action, worker) in zip(ego_actions, op_actions, self.workers):	        
                res_obs.append(worker.step.remote([ego_action, op_action]))
        else:
            for num, worker in enumerate(self.workers):
                actions_per_env = []
                actions_per_env.append(ego_actions[self.num_agents * num: self.num_agents * num + self.num_agents])
                actions_per_env.append(op_actions[self.num_agents * num: self.num_agents * num + self.num_agents])
                res_obs.append(worker.step.remote(actions_per_env))

        all_res = ray.get(res_obs)
        for res in all_res:
            cobs, crewards, cdones, cinfos = res
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newobs_op.append(cobs['obs_op'])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs['obs'])
                newobs_op.append(cobs['obs_op'])

            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        newobsdict = {}
        newobsdict["obs"] = self.concat_func(newobs)
        newobsdict["obs_op"] = self.concat_func(newobs_op)

        if self.use_global_obs:
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)          
        ret_obs = newobsdict
        if self.concat_infos:
            newinfos = dicts_to_dict_with_arrays(newinfos, False)
        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

    def get_env_info(self):
        res = self.workers[0].get_env_info.remote()
        return ray.get(res)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self, init=False):
        res_obs = [worker.reset.remote(init) for worker in self.workers]
        newobs, newobs_op, newstates = [], [], []
        for res in res_obs:
            cobs = ray.get(res)
            if self.use_global_obs:
                newstates.append(cobs["state"])
            newobs.append(cobs["obs"])
            newobs_op.append(cobs['obs_op'])

        newobsdict = {}
        newobsdict["obs"] = self.concat_func(newobs)
        newobsdict["obs_op"] = self.concat_func(newobs_op)

        if self.use_global_obs:            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)      
      
        ret_obs = newobsdict
        return ret_obs


class AMPSPRayVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        motion_file = kwargs['motion_file']
        imitate_type = kwargs.get('imitate_type', "s_s")
        self._load_motion(motion_file, imitate_type)
        self.seeds = kwargs.pop('seed', None)
        self.remote_worker = ray.remote(SPRayWorker)
        worker_index = kwargs.get("worker_index", 888)
        self.workers = [self.remote_worker.remote(self.config_name, kwargs, i+worker_index) for i in range(self.num_actors)]

        if self.seeds is not None:
            seeds = range(self.seeds, self.seeds + self.num_actors)
            seed_set = []
            for (seed, worker) in zip(seeds, self.workers):	        
                seed_set.append(worker.seed.remote(seed))
            ray.get(seed_set)

        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = ray.get(res)

        res = self.workers[0].get_env_info.remote()
        env_info = ray.get(res)
        res = self.workers[0].can_concat_infos.remote()
        can_concat_infos = ray.get(res)
        self.use_global_obs = env_info['use_global_observations']
        self.concat_infos = can_concat_infos
        self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
        self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
        if self.num_agents == 1:
            self.concat_func = np.concatenate
        else:
            self.concat_func = np.concatenate

        # AMP
        self._num_amp_obs_steps = kwargs.get("numAMPObsSteps", 10)
        assert(self._num_amp_obs_steps >= 2)

        # features only about pose and low-level control
        self._num_amp_obs_per_step = 100
        self.amp_observation_space = spaces.Box(np.ones(self.get_num_amp_obs()) * -np.Inf, np.ones(self.get_num_amp_obs()) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_actors, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None


    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)

        amp_obs_demo = self.build_amp_obs_demo(num_samples * self._num_amp_obs_steps)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def build_amp_obs_demo(self, num_samples):
        amp_obs_demo = self._motion_lib.sample(num_samples)


    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), dtype=torch.float32)
        return


    def _load_motion(self, motion_file, imitate_type):
        self._motion_lib = MotionLib(motion_file, imitate_type)
        return



    def step(self, actions):
        newobs, newobs_op, newstates, newrewards, newdones, newinfos = [], [], [], [], [], []
        ego_actions, op_actions = actions
        res_obs = []
        if self.num_agents == 1:
            for (ego_action, op_action, worker) in zip(ego_actions, op_actions, self.workers):	        
                res_obs.append(worker.step.remote([ego_action, op_action]))
        else:
            for num, worker in enumerate(self.workers):
                actions_per_env = []
                actions_per_env.append(ego_actions[self.num_agents * num: self.num_agents * num + self.num_agents])
                actions_per_env.append(op_actions[self.num_agents * num: self.num_agents * num + self.num_agents])
                res_obs.append(worker.step.remote(actions_per_env))

        all_res = ray.get(res_obs)
        for res in all_res:
            cobs, crewards, cdones, cinfos = res
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newobs_op.append(cobs['obs_op'])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs['obs'])
                newobs_op.append(cobs['obs_op'])

            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        newobsdict = {}
        newobsdict["obs"] = self.concat_func(newobs)
        newobsdict["obs_op"] = self.concat_func(newobs_op)

        if self.use_global_obs:
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)          
        ret_obs = newobsdict
        if self.concat_infos:
            newinfos = dicts_to_dict_with_arrays(newinfos, False)
        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

    def get_env_info(self):
        res = self.workers[0].get_env_info.remote()
        env_info = ray.get(res)
        env_info['amp_observation_space'] = self.amp_observation_space

        return env_info

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self, init=False):
        res_obs = [worker.reset.remote(init) for worker in self.workers]
        newobs, newobs_op, newstates = [], [], []
        for res in res_obs:
            cobs = ray.get(res)
            if self.use_global_obs:
                newstates.append(cobs["state"])
            newobs.append(cobs["obs"])
            newobs_op.append(cobs['obs_op'])

        newobsdict = {}
        newobsdict["obs"] = self.concat_func(newobs)
        newobsdict["obs_op"] = self.concat_func(newobs_op)

        if self.use_global_obs:            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)      
      
        ret_obs = newobsdict
        return ret_obs
