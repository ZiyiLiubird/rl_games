import socket
import json

class env:

    IP='127.0.1.1'
    PORT=8888
    INITIAL = False
    RENDER = 0

    def __init__(self, ip: str=IP, port: int=PORT, render: int=RENDER):
        self.IP = ip
        self.PORT = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.IP, self.PORT))
        self.RENDER = render

    def _send_condition(self, data):
        self.socket.send(bytes(data.encode('utf-8')))

    def _accept_from_socket(self):

        msg_receive = json.loads(str(self.socket.recv(50000), encoding='utf-8'))

        return msg_receive

    def reset(self, red_number:int, blue_number: int, reset_attribute:dict):
        init_flag = ""
        init_info = {'flag': {}, 'red': {}, 'blue': {}}
        if self.INITIAL == False:
            self.INITIAL = True
            init_flag = "init"
            init_info['flag'] = {'init': {'render': self.RENDER}}
        else:
            init_flag = "reset"
            init_info['flag'] = {'reset': {}}

        for number in range(red_number):
            try:
                init_info['red']['red_'+str(number)]=reset_attribute['red']['red_'+str(number)]
            except:
                print ('load red plane attribute error, please check the form of initial attribute dictionary')

        for number in range(blue_number):
            try:
                init_info['blue']['blue_' + str(number)] = reset_attribute['blue']['blue_' + str(number)]
            except:
                print('load blue plane attribute error, please check the form of initial attribute dictionary')

        data = json.dumps(init_info)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()
        return msg_receive

    def reset(self, reset_attribute):

        data = json.dumps(reset_attribute)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()
        return msg_receive

    def step(self, action_attribute):

        data = json.dumps(action_attribute)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()

        return msg_receive

    def get_obs(self):

        init_info = {'flag':{'obs':{}}}
        data = json.dumps(init_info)
        self._send_condition(data)
        msg_receive = self._accept_from_socket()
        return  msg_receive


    def get_obs_red(self):
        global_msg = self.get_obs()
        red_msg = global_msg['red']
        blueNum = set()
        for red_Fighter in red_msg.values():
            m_TargetNum = str(int(red_Fighter["TargetIntoView"]))  # 进入视野内的敌机编号
            if m_TargetNum != "99":
                for num in m_TargetNum:
                    blueNum.add(num)
        for num in blueNum:
            red_msg['blue_' + num] = global_msg['blue']['blue_' + num]
        return red_msg

    def get_obs_blue(self):

        global_msg = self.get_obs()
        blue_msg = global_msg['blue']
        redNum = set()
        for red_Fighter in blue_msg.values():
            m_TargetNum = str(int(red_Fighter["TargetIntoView"]))  # 进入视野内的敌机编号
            if m_TargetNum != "99":
                for num in m_TargetNum:
                    redNum.add(num)
        for num in redNum:
            blue_msg['red_' + num] = global_msg['red']['red_' + num]
        return blue_msg


