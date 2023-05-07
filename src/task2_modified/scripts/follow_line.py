import math
import numpy as np

class FollowLine(object):
    def __init__(self, max_speed: float, length: float, r_length: float, start_point: tuple, target_point: np.ndarray):
        self._max_speed = max_speed
        self._length = length
        self._r_length = r_length
        self._start_point = start_point
        self._point_set = self.generate_path()
        self._target_point_position = self.get_target_point_pos(target_point)
        self._len_point = self._point_set.shape[0]
        self._Kv = 0.9
        self._Kp = 0.2
        self._car_Kv = 0.95
        self._car_Kp = 0.2
        self._visit_position = 0

    def generate_path(self) -> np.ndarray:
        delta = 0.1
        point_set = []
        for i in range(int(self._length / delta) + 1):
            point_set.append(np.array([self._start_point[0], self._start_point[1]+i*delta]))

        return np.array(point_set)

    def get_target_point_pos(self, target_point):
        _distance = 1e3
        temp_position = 0
        for i in range(self._len_point):
            distance = np.linalg.norm(target_point - self._point_set[i])
            if _distance > distance:
                _distance = distance
                temp_position = i
        return temp_position

    def get_car_velocity(self, car_pos: np.ndarray, obj_pos: np.ndarray):
        car0_pos = car_pos[0, :]
        car1_pos = car_pos[1, :]
        car2_pos = car_pos[2, :]
        car3_pos = car_pos[3, :]
        obj_2_car0_vector = np.array([self._r_length, self._r_length]) / math.sqrt(2)
        obj_2_car1_vector = np.array([-self._r_length, self._r_length]) / math.sqrt(2)
        obj_2_car2_vector = np.array([-self._r_length, -self._r_length]) / math.sqrt(2)
        obj_2_car3_vector = np.array([self._r_length, -self._r_length]) / math.sqrt(2)
        car0_pos_next_time = obj_pos + obj_2_car0_vector
        car1_pos_next_time = obj_pos + obj_2_car1_vector
        car2_pos_next_time = obj_pos + obj_2_car2_vector
        car3_pos_next_time = obj_pos + obj_2_car3_vector
        obj_cur_vel = self.get_obj_velocity(obj_pos)

        car0_cur_vel = obj_cur_vel
        car1_cur_vel = obj_cur_vel
        car2_cur_vel = obj_cur_vel
        car3_cur_vel = obj_cur_vel

        car0_cur_vel = self._car_Kv * car0_cur_vel + self._car_Kp * (car0_pos_next_time - car0_pos)
        car1_cur_vel = self._car_Kv * car1_cur_vel + self._car_Kp * (car1_pos_next_time - car1_pos)
        car2_cur_vel = self._car_Kv * car2_cur_vel + self._car_Kp * (car2_pos_next_time - car2_pos)
        car3_cur_vel = self._car_Kv * car3_cur_vel + self._car_Kp * (car3_pos_next_time - car3_pos)
        car_cur_vel = np.concatenate((car0_cur_vel, car1_cur_vel, car2_cur_vel, car3_cur_vel))

        return car_cur_vel

    def get_obj_velocity(self, obj_pos: np.ndarray):
        cur_vel = np.array([0, self._max_speed])
        cur_vel = self._Kv * cur_vel + self._Kp * (self.get_nearest_line_point(obj_pos) - obj_pos)
        return cur_vel

    def get_nearest_line_point(self, obj_pos: np.ndarray):
        _distance = 1e3
        temp_position = self._visit_position
        start_position = self._visit_position
        end_position = start_position + 10
        if end_position >= self._len_point - 1:
            end_position = self._len_point - 1
        for i in range(start_position, end_position):
            distance = np.linalg.norm(obj_pos-self._point_set[i])
            if _distance > distance:
                _distance = distance
                temp_position = i

        self._visit_position = temp_position
        p_next_time = self._point_set[self._visit_position + 1]

        return p_next_time

    def second_replace_visit_position(self):
        self._visit_position = self._out_point_position

