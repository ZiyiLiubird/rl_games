import time

import comm_interface

# 多对多 数据交互模板
# 单位参考AI接口说明文档
# 其中 ic/long-gc-deg  初始经度
#     ic/lat-geod-deg 初始维度
#     0,0代表处于中心  * 110574 代表unity中 米单位的世界坐标

# 发送初始化数据   render 0 训练  1演示
# 注：飞机数量与unity设置保持一致
# 飞机名称 统一为 red_0  red_1 ...

dict_init = {'flag':{'init':{'render': 0}},
             'red': {
                 'red_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.01,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_1': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.05,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.1,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.15,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.2,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                'red_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.25,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
             },
             'blue': {
                 'blue_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.01,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_1': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.05,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.1,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.15,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.2,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.25,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
             }}

# 发送重置数据 可与初始化数据不一致
dict_reset = {'flag':{'reset':{}},
             'red': {
                 'red_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.01,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_1': {"ic/h-sl-ft": 8000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.05,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.1,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.15,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                 'red_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.2,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                'red_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.1,
                          "ic/lat-geod-deg": 0.25,
                          "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                          "ic/r-rad_sec": 0,
                          "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
             },
             'blue': {
                 'blue_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.01,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_1': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.05,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.1,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.15,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.2,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                 'blue_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.1,
                           "ic/lat-geod-deg": 0.25,
                           "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                           "ic/r-rad_sec": 0,
                           "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
             }}

# 实时发送各飞机的控制指令
# fcs/aileron-cmd-norm  fcs/elevator-cmd-norm  fcs/rudder-cmd-norm  取值范围  -1~1
# fcs/throttle-cmd-norm 取值范围 0~1
# fcs/weapon-launch  0 不发射  1发射导弹  2发射子弹
# change-target  瞄准目标编号  99保持不变 88由程序中按进入攻击区顺序锁定(根据导弹类型不同 1目标 1~4目标)  0123(表示锁定 对方编号0、1、2、3四架飞机)(当目标进入攻击区优先锁定)
command_sent = {'red':{'red_0':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                       'red_1':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                       'red_2':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                       'red_3':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                       'red_4':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                       'red_5':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0}
                       },
                'blue':{'blue_0':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        'blue_1':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        'blue_2':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        'blue_3':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        'blue_4':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        'blue_5':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0}
                       }
                }

# 测试加速所用
num = 0
start_time = 0
current_time = 0

# 初始化通信信息
# ip
# port
# 模式  编写训练代码时使用 需根据自己代码进行修改
env=comm_interface.env('127.0.0.1',8888,0)
# 设置发送初始化时的模式
# 编写训练代码时使用 需根据自己代码进行修改
# env.RENDER = 1 #  0 训练  1演示

# 发送初始化数据  接收生成飞机的所有数据
# 编写训练代码时使用 需根据自己代码进行修改
# msg_reieve = env.reset(1, 1, dict_init)

#  测试时 直接发送初始化数据
msg_reieve = env.reset(dict_init)

# 测试加速倍速所用
start_time = time.time()
red1_sate = 1000000
blue1_sate = 1000000

while(1):

    if  red1_sate < 10000:
        zx_time = (current_time - start_time) * 1000
        bs = ((num-1) * 200) / zx_time
        print("次数：" + str(num) + " 耗时：" + str(zx_time) + " 加速倍速：" + str(bs))
        start_time = time.time()
        # 发送重置指令 并接收数据
        # 编写训练代码时使用 需根据自己代码进行修改
        # _ = env.reset(1, 1, dict_reset)

        # 测试直接发送重置
        _ = env.reset(dict_reset)
        num = 0

    # 排除第一次生成飞机消耗时间
    if num==1:
        start_time = time.time()
    # 实时发送各飞机指令
    state = env.step(command_sent)
    # print(state)
    # 获取对应数据模板
    red1_sate = state['red']['red_0']['position/h-sl-ft']
    num = num + 1
    current_time = time.time()

    # 获取导演视角下，当前时刻各个飞机状态 并不会推动程序执行   只有env.step指令 unity会根据指令执行
    # obs_state = env.get_obs()

    #获取红方视角下，当前时刻飞机状态，包含红方所有战机信息与蓝方被侦察到战机的信息
    # obs_red = env.get_obs_red()
    # # obs_blue = env.get_obs_blue()
