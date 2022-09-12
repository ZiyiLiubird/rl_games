import os
from os.path import dirname
import sys

main_path = dirname(dirname(os.path.abspath(__file__)))
print(f"{main_path}")
sys.path.append(main_path)
import numpy as np

from zhikong.AirCombat_Env import AirCombatEnv

exp_name = "aircombat_env"

config = {"ip": '127.0.1.1', "num_agents":2, "worker_index":888, "scenes":1,
          'excute_path': "/home/lzy/lzy/human-ai/Air/2.4/Linux/ZK.x86_64",
          "playmode": 1, "action_space_type": "MultiDiscrete", "red_agents_num":1,
          "blue_agents_num":1, "limit": 100000}

env = AirCombatEnv(**config)
obs = env.reset(init=True)

ego_actions = np.array([[0,0.5,0,1,0,0]], dtype=float)
op_actions = np.array([[0,0,0,1,0,0]], dtype=float)


while True:
    actions = [ego_actions, op_actions]
    obs, rewards, dones, infos = env.step(actions)
    if np.all(dones):
        print(f"episode done!")
        obs = env.reset()

