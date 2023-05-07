import numpy as np
import random
import matplotlib.pyplot as plt
import mujoco as mp
import numpy as np
import math
import os, sys


def quart_to_rpy(w: float, x: float, y: float, z: float):
    roll = math.atan2(2 * (y * x + w * z), 1 - 2 * (z * z + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return (roll, pitch, yaw)


def plot_y(data, label_list: list) -> None:
    n = len(data)
    assert len(data) == len(label_list)
    for i in range(n):
        plt.plot(np.arange(0, len(data[i]), 1), data[i], label=label_list[i])
    plt.legend()
    plt.show()

# 欧拉角转换为四元数, 旋转顺序为ZYX(偏航角yaw, 俯仰角pitch, 横滚角roll)
def eular2quat(yaw, pitch, roll):
    # 注意这里必须先转换为弧度, 因为这里的三角计算均使用的是弧度.
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    # 笛卡尔坐标系
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

class car_mujoco_env():
    def __init__(self, render=False, debug=False, train=True, random_move=False, mode=1):
        if train:
            self._model = mp.MjModel.from_xml_path(os.path.abspath(os.path.join(sys.path[0], "../mujoco_model/omni_robot_reach_target_train.xml")))
        else:
            if mode == 1: # move line env
                self._model = mp.MjModel.from_xml_path(os.path.abspath(os.path.join(sys.path[0], "../mujoco_model/omni_robot_reach_target_test_line.xml")))
            elif mode == 2:
                self._model = mp.MjModel.from_xml_path(os.path.abspath(os.path.join(sys.path[0], "../mujoco_model/omni_robot_reach_target_test_eight.xml")))
            elif mode == 3:
                self._model = mp.MjModel.from_xml_path(os.path.abspath(os.path.join(sys.path[0], "../mujoco_model/omni_robot_reach_target_test_tanh.xml")))

        self._sim = mp.MjSim(self._model)
        if render:
            self._viewer = mp.MjViewer(self._sim)
        self._render = render
        self._debug = debug
        self._each_step = 50
        self._random_move = random_move
        self._cnt = 0
    
    def get_sensor(self, name, dimensions):
        i = self._sim.model.sensor_name2id(name)
        return self._sim.data.sensordata[i:i+dimensions]    
    
    def get_target_object_roll_angle(self):
        car_xquat = self._sim.data.get_body_xquat("target_object")
        roll, pitch, yaw = quart_to_rpy(car_xquat[0], car_xquat[1], car_xquat[2], car_xquat[3])
        angle = np.array([roll, pitch, yaw])
        angle = angle / 3.14 * 180
        for i in range(3):
            if angle[i] < 0:
                angle[i] += 360
        return angle
    
    def get_car_angle(self):
        car_ang = []
        car_xquat = self._sim.data.get_body_xquat("omni_car")
        roll, pitch, yaw = quart_to_rpy(car_xquat[0], car_xquat[1], car_xquat[2], car_xquat[3])
        car_ang.append(roll)
        car_xquat = self._sim.data.get_body_xquat("omni_car2")
        roll, pitch, yaw = quart_to_rpy(car_xquat[0], car_xquat[1], car_xquat[2], car_xquat[3])
        car_ang.append(roll)
        car_xquat = self._sim.data.get_body_xquat("omni_car3")
        roll, pitch, yaw = quart_to_rpy(car_xquat[0], car_xquat[1], car_xquat[2], car_xquat[3])
        car_ang.append(roll)
        car_xquat = self._sim.data.get_body_xquat("omni_car4")
        roll, pitch, yaw = quart_to_rpy(car_xquat[0], car_xquat[1], car_xquat[2], car_xquat[3])
        car_ang.append(roll)
        return np.array(car_ang)
    
    def get_target_object_pos(self):
        return self._sim.data.get_body_xpos("target_object")[:2]
    
    def get_target_object_height(self):
        return np.array([self._sim.data.get_body_xpos("target_object")[2]])
            
    def set_target_ref_pos(self, pos):
        self._sim.data.mocap_pos[0][0:2] = pos[:]
    
    def set_target_ref_angle(self, eular):
        # eular = (eular - 180) * -1
        quat = eular2quat(eular, 0, 0)
        self._sim.data.mocap_quat[0] = quat
    
    def set_obstacle_pos(self, pos: list, radius, height) -> None:
        self._sim.data.mocap_pos[1][0:2] = pos[:]
        self._sim.data.mocap_pos[1][2] = height[:]
        self._model.geom_size[1][0] = radius
        self._model.geom_size[1][1] = height
    
    def get_obstacle_pos(self) -> np.ndarray:
        return self._sim.data.mocap_pos[1][0:2]
    
    def get_obstacle_radius(self) -> float:
        return self._model.geom_size[1][0]
    
    def set_car_target_pos(self, car_target_pos: np.ndarray) -> None:
        self._sim.data.mocap_pos[2][0:2] = car_target_pos[0:2]
        self._sim.data.mocap_pos[3][0:2] = car_target_pos[2:4]
        self._sim.data.mocap_pos[4][0:2] = car_target_pos[4:6]
        self._sim.data.mocap_pos[5][0:2] = car_target_pos[6:8]

    def get_car_pos(self):
        pos =np.zeros((4, 2))
        pos[0] = self._sim.data.get_body_xpos("omni_car")[:2]
        pos[1] = self._sim.data.get_body_xpos("omni_car2")[:2]
        pos[2] = self._sim.data.get_body_xpos("omni_car3")[:2]
        pos[3] = self._sim.data.get_body_xpos("omni_car4")[:2]

        return pos
    
    def random_init(self):
        if self._random_move:
            n = random.randint(0, 8)
            for _ in range(n):
                speed = np.random.uniform(-1, 1, 8)
                self.control(speed)

    def reset(self):
        self._sim.reset()
        self.random_init()
    
    # [x_speed, y_speed, x_speed, y_speed, x_speed, y_speed, x_speed, y_speed]
    def control(self, speed):
        self._cnt += 1
        leng = self._cnt % 20
        direct = -1 if leng > 4 else 1
        # self.set_obstacle_pos(self.get_obstacle_pos() + 0.001 * leng * direct, self.get_obstacle_radius())
        for _ in range(self._each_step):
            self._sim.data.ctrl[:] = speed
            if self._render:
                self._viewer.render()

            try:
                self._sim.step()     
            except:
                return False
        return True

def test():
    omni_robot = car_mujoco_env(True,True)
    for _ in range(1000):
        omni_robot.control(np.random.uniform(-0.2, 0.2, 8))

if __name__ == '__main__':
    test()
