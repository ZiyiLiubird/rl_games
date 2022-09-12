import time

import comm_interface

# 1架飞机数据交互模板
# 单位参考AI接口说明文档
# 其中 ic/long-gc-deg  初始经度
#     ic/lat-geod-deg 初始维度
#     0,0代表处于中心  * 110574 代表unity中 米单位的世界坐标

# 发送初始化数据 render 0 训练  1演示 对应飞机阵营 名称 需与unity设置保持一致
# 注：飞机数量与unity设置保持一致
# 飞机名称 统一为 red_0  red_1 ...
dict_init = {'flag':{'init':{'render': 1}},'red':{'red_0':
{"ic/h-sl-ft": 50000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.01, "ic/lat-geod-deg": 0.01,
 "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
 "ic/roc-fpm": 0, "ic/psi-true-deg": 135}}}

# 发送重置数据  可与初始化数据不一致
dict_reset = {'flag':{'reset':{}},'red':{'red_0':
{"ic/h-sl-ft": 50000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.01, "ic/lat-geod-deg": 0.01,
 "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
 "ic/roc-fpm": 0, "ic/psi-true-deg": 135}}}

# 实时发送各飞机的控制指令
# fcs/aileron-cmd-norm  fcs/elevator-cmd-norm  fcs/rudder-cmd-norm  取值范围  -1~1
# fcs/throttle-cmd-norm 取值范围 0~1
# fcs/weapon-launch  0 不发射  1发射导弹  2发射子弹
# change-target  00(不变)
#                99(由程序进行控制)
#                0\1\12\012\0134（优先锁定目标机编号-可设置1~4架）
command_sent = {'red':{'red_0':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0}}}

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
# env.RENDER = 1 # 0 训练  1演示

# 发送初始化数据  接收生成飞机的所有数据
# 编写训练代码时使用 需根据自己代码进行修改
# msg_reieve = env.reset(1, 1, dict_init)

#  测试时 直接发送初始化数据
msg_reieve = env.reset(dict_init)
red1_time = msg_reieve['red']['red_0']['simulation/sim-time-sec']
print(red1_time)
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
    # 获取对应数据模板
    red1_sate = state['red']['red_0']['position/h-sl-ft']
    red1_time = state['red']['red_0']['simulation/sim-time-sec']
    print(red1_time)
    num = num + 1
    current_time = time.time()





# obs_state=env.get_obs()
# obs_red=env.get_obs_red()
# obs_blue=env.get_obs_blue()
#
# print('reset_msg:{0}',msg_reieve)
# print('next_state:{0}',state)
# print('obs_state:{0}',obs_state)
# print('obs_red:{0}',obs_red)
# print('obs_blue:{0}',obs_blue)