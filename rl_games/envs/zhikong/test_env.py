import os
from os.path import dirname
import sys

main_path = dirname(dirname(os.path.abspath(__file__)))
print(f"{main_path}")
sys.path.append(main_path)
import numpy as np

from zhikong.AirCombat_Env import AirCombatEnv

exp_name = "aircombat_env"

# /home/lzy/lzy/human-ai/Air/8-23/2.4/Linux/Mono/Linux
# /home/lzy/lzy/human-ai/Air/2.4/Linux/ZK.x86_64

config = {"ip": '127.0.1.1', "num_agents":6, "worker_index":888, "scenes":1,
          'excute_path': "/home/lzy/lzy/human-ai/Air/8-23/2.4/Linux/Mono/Linux/ZK.x86_64",
          "playmode": 1, "action_space_type": "MultiDiscrete", "red_agents_num":3,
          "blue_agents_num":3, "episode_max_length": 1000, "change_target": False}

env = AirCombatEnv(**config)
obs = env.reset(init=True)

# 1v1
# ego_actions = np.array([[0,0,0,1,0,0]], dtype=float)
# op_actions = np.array([[0,0,0,1,0,0]], dtype=float)

# 2v2
# ego_actions = np.array([[0,0,0,1,0,0], [0,0,0,1,0,0]], dtype=float)
# op_actions = np.array([[0,0,0,1,0,0], [0,0,0,1,0,0]], dtype=float)

# 3v3
ego_actions = np.array([[4,4,4,4,1,0], [4,4,4,4,1,0], [4,4,4,4,1,0]], dtype=int)
op_actions = np.array([[4,4,4,4,0,0], [4,4,4,4,0,0], [4,4,4,4,0,0]], dtype=int)



steps = 0
while True:
    actions = [ego_actions, op_actions]
    obs, rewards, dones, infos = env.step(actions)
    steps += 1
    print(f"steps: {steps}")
    if np.all(dones):
        print(f"episode done!")
        obs = env.reset()
        steps = 0

