import os
from os.path import dirname
import sys
import torch

main_path = dirname(dirname(os.path.abspath(__file__)))
print(f"{main_path}")
sys.path.append(main_path)
import numpy as np
from collections import namedtuple
from zhikong.AirCollector_Env import AirCollectEnv


def save_buffer(path, obs_buffer, action_buffer, obs_next_ubffer):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    data = {}
    data['obs'] = obs_buffer
    data['actions'] = action_buffer
    data['obs_next'] = obs_next_buffer
    torch.save(data, path + '.pth')
    print(f"save success !!!")




if __name__ == "__main__":

    # setting 0: 相对飞行
    # setting 1: 同向追击

    config = {"ip": '127.0.1.1', "num_agents":6, "worker_index":888, "scenes":1,
             'excute_path': '/home/lzy/lzy/human-ai/Air/9-21/Linux/Mono/Linux/ZK.x86_64',
             "playmode": 1, "action_space_type": "MultiDiscrete", "red_agents_num":3,
             "blue_agents_num":3, "episode_max_length": 500, "change_target": False,
             "setting": 0, "enemy_weapon": 1,}

    env = AirCollectEnv(**config)
    obs = env.reset(init=True)

    # 1v1
    ego_actions = np.array([[0,0,1,0,0]], dtype=float)
    op_actions = np.array([[0,0,1,0,0]], dtype=float)


    transition = namedtuple('transition', ['obs', 'action', 'obs_next'])
    buffer = []
    obs_buffer = []
    action_buffer = []
    obs_next_buffer = []

    game_nums = 20
    path = "/home/lzy/lzy/MARL/self-play/data/data"
    steps = 0
    episode = 0
    while True:
        obs_buffer.append(obs)
        
        actions = [ego_actions, op_actions]
        obs_next, actions_human, dones = env.step(actions)

        action_buffer.append(actions_human)
        obs_next_buffer.append(obs_next)
        obs = obs_next

        steps += 1
        print(f"steps: {steps}")
        if np.all(dones):
            print(f"episode done!")
            obs = env.reset()
            steps = 0
            episode += 1

        if episode >= game_nums:
            break

    save_buffer(path, obs_buffer=obs_buffer, action_buffer=action_buffer,
                obs_next_ubffer=obs_next_buffer)