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
    # np.savez(path, obs_buffer=obs_buffer,
    #          action_buffer=action_buffer, obs_next_buffer=obs_next_buffer)




if __name__ == "__main__":

    config = {"ip": '127.0.1.1', "num_agents":2, "worker_index":888, "scenes":1,
            'excute_path': '/home/lzy/lzy/human-ai/Air/9-17/Linux/IL2CPP/Linux/ZK.x86_64',
            "playmode": 1, "action_space_type": "MultiDiscrete", "red_agents_num":1,
            "blue_agents_num":1, "episode_max_length": 500, "change_target": False}

    env = AirCollectEnv(**config)
    obs = env.reset(init=True)

    # 1v1
    ego_actions = np.array([[4,4,4,4,0,0]], dtype=float)
    op_actions = np.array([[4,4,4,4,0,0]], dtype=float)

    transition = namedtuple('transition', ['obs', 'action', 'obs_next'])
    buffer = []
    obs_buffer = []
    action_buffer = []
    obs_next_buffer = []

    game_nums = 1
    path = "/home/lzy/lzy/MARL/self-play/data/data"
    steps = 0
    episode = 0
    while True:
        actions = [ego_actions, op_actions]
        obs_next, actions_human, dones = env.step(actions)

        buffer.append(transition(obs=obs, action=actions_human, obs_next=obs_next))
        obs_buffer.append(obs)
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