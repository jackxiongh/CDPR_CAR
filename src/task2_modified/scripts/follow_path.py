from typing import Tuple
import numpy as np

class FollowPath(object):
    def __init__(self, start_point: np.ndarray, path_size: np.ndarray, 
        max_speed: float, r_length: float, delta: float, window: float, Kv: float, Kp: float,
        car_Kv: float, car_Kp: float)-> None:       
        # start_point: the path is relative to the start point
        # path_size: size of the path, different in different shapes
        # max_speed: max speed of the cars
        # r_length: distance between cars
        # delta: the distance between two way points (in meter)
        # window: searching window (in meter)
        self._start_point = start_point
        self._max_speed = max_speed
        self._r_length = r_length
        self._delta = delta
        self._Kv = Kv
        self._Kp = Kp
        self._car_Kv = car_Kv
        self._car_Kp = car_Kp
        self._window = window
        self._point_set = self.generate_path()
        self._len_point = self._point_set.shape[0]
        self._visit_position = 0

        # translation from the object to the car
        self.obj_2_car0_vector = (np.array([self._r_length, self._r_length]) / 2).reshape((1, 2))
        self.obj_2_car1_vector = (np.array([-self._r_length, self._r_length]) / 2).reshape((1, 2))
        self.obj_2_car2_vector = (np.array([-self._r_length, -self._r_length]) / 2).reshape((1, 2))
        self.obj_2_car3_vector = (np.array([self._r_length, -self._r_length]) / 2).reshape((1, 2))

    def get_nearest_point(self, obj_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _distance = 1e3
        temp_position = self._visit_position
        start_position = self._visit_position
        end_position = start_position + int(self._window / self._delta)
        if end_position >= self._len_point - 1:
            end_position = self._len_point - 1
        for i in range(start_position, end_position):
            distance = np.linalg.norm(obj_pos-self._point_set[i])
            if _distance > distance:
                _distance = distance
                temp_position = i

        self._visit_position = temp_position

        return self._point_set[self._visit_position], self._point_set[self._visit_position + 2]

    def get_car_velocity(self, car_pos: np.ndarray, obj_pos: np.ndarray, obj_ang: float, car_ang: np.ndarray) -> np.ndarray:
        # car_pos: 4x2 np.ndarray
        # obj_pos: 1x2 np.ndarray
        # obj_ang: object angle relative to the world frame
        # car_ang: car angle relative to the object frame
        # return: 1x8 np.ndarray

        # get car position
        car0_pos = (car_pos[0, :]).reshape((1, 2))
        car1_pos = (car_pos[1, :]).reshape((1, 2))
        car2_pos = (car_pos[2, :]).reshape((1, 2))
        car3_pos = (car_pos[3, :]).reshape((1, 2))

        # get next point of the nearest point
        obj_nearest_pos, obj_next_pos = self.get_nearest_point(obj_pos)
        obj_next_pos = obj_next_pos.reshape((1, 2))

        # p_i = p_0 + R * p_i0
        # expected angle of the object
        obj_exp_ang = self.get_obj_angle(obj_next_pos[0])
        rot = np.array([[np.cos(obj_exp_ang), -np.sin(obj_exp_ang)], 
                        [np.sin(obj_exp_ang), np.cos(obj_exp_ang)]])
        car0_pos_next_time = (obj_next_pos.transpose() + rot @ self.obj_2_car0_vector.transpose()).transpose()
        car1_pos_next_time = (obj_next_pos.transpose() + rot @ self.obj_2_car1_vector.transpose()).transpose()
        car2_pos_next_time = (obj_next_pos.transpose() + rot @ self.obj_2_car2_vector.transpose()).transpose()
        car3_pos_next_time = (obj_next_pos.transpose() + rot @ self.obj_2_car3_vector.transpose()).transpose()

        # v_i = v_0 + omega x obj2car
        obj_cur_vel = self.get_obj_velocity(obj_pos).reshape((1, 2))
        obj_cur_vel = self._Kv * obj_cur_vel + self._Kp * (obj_next_pos - obj_pos.reshape((1, 2)))
        # rotate the velocity into the object frame
        rot = np.array([[np.cos(obj_ang), np.sin(obj_ang)], 
                        [-np.sin(obj_ang), np.cos(obj_ang)]])
        obj_cur_vel = (rot @ obj_cur_vel.transpose()).transpose()
        omega = self.get_omega(obj_pos)
        car0_cur_vel = obj_cur_vel + np.array([[-omega * self.obj_2_car0_vector[0, 1], omega * self.obj_2_car0_vector[0, 0]]])
        car1_cur_vel = obj_cur_vel + np.array([[-omega * self.obj_2_car1_vector[0, 1], omega * self.obj_2_car1_vector[0, 0]]])
        car2_cur_vel = obj_cur_vel + np.array([[-omega * self.obj_2_car2_vector[0, 1], omega * self.obj_2_car2_vector[0, 0]]])
        car3_cur_vel = obj_cur_vel + np.array([[-omega * self.obj_2_car3_vector[0, 1], omega * self.obj_2_car3_vector[0, 0]]])

        # v = Kv * v_target + Kp * (p_target - p_current)
        car0_cur_vel = self._car_Kv * car0_cur_vel + self._car_Kp * (rot @ ((car0_pos_next_time - car0_pos).transpose())).transpose()
        car1_cur_vel = self._car_Kv * car1_cur_vel + self._car_Kp * (rot @ ((car1_pos_next_time - car1_pos).transpose())).transpose()
        car2_cur_vel = self._car_Kv * car2_cur_vel + self._car_Kp * (rot @ ((car2_pos_next_time - car2_pos).transpose())).transpose()
        car3_cur_vel = self._car_Kv * car3_cur_vel + self._car_Kp * (rot @ ((car3_pos_next_time - car3_pos).transpose())).transpose()

        rot = np.array([[np.cos(car_ang[0]), np.sin(car_ang[0])], 
                        [-np.sin(car_ang[0]), np.cos(car_ang[0])]])
        car0_cur_vel = (rot @ car0_cur_vel.transpose()).transpose()
        rot = np.array([[np.cos(car_ang[1]), np.sin(car_ang[1])], 
                        [-np.sin(car_ang[1]), np.cos(car_ang[1])]])
        car1_cur_vel = (rot @ car1_cur_vel.transpose()).transpose()
        rot = np.array([[np.cos(car_ang[2]), np.sin(car_ang[2])], 
                        [-np.sin(car_ang[2]), np.cos(car_ang[2])]])
        car2_cur_vel = (rot @ car2_cur_vel.transpose()).transpose()
        rot = np.array([[np.cos(car_ang[3]), np.sin(car_ang[3])], 
                        [-np.sin(car_ang[3]), np.cos(car_ang[3])]])
        car3_cur_vel = (rot @ car3_cur_vel.transpose()).transpose()
        
        return np.concatenate((car0_cur_vel[0], car1_cur_vel[0], car2_cur_vel[0], car3_cur_vel[0])), \
                np.concatenate((car0_pos_next_time[0], car1_pos_next_time[0], car2_pos_next_time[0], car3_pos_next_time[0]))
    
    def get_target_point(self, obstacle_pos: np.ndarray, obstacle_radius: np.ndarray) -> np.ndarray:
        # obstacle_pos: 1x2 np.ndarray
        # obstacle_radius: 1x1 np.ndarray
        # return: 1x2 np.ndarray
        target_pos = []
        if_in_obstacle = False
        for i in range(self._len_point):
            if (not if_in_obstacle) and np.linalg.norm(self._point_set[i] - obstacle_pos) < obstacle_radius:
                if_in_obstacle = True
            elif if_in_obstacle and np.linalg.norm(self._point_set[i] - obstacle_pos) > obstacle_radius:
                if_in_obstacle = False
                target_pos.append(self._point_set[i])
        return np.array(target_pos)
            
    def generate_path(self) -> np.ndarray:
        pass

    def get_obj_velocity(self, obj_pos: np.ndarray) -> np.ndarray:
        pass

    def get_omega(self, obj_pos: np.ndarray) -> float:
        pass

    def get_obj_angle(self, obj_pos: np.ndarray) -> float:
        pass

class FollowEight(FollowPath):
    def __init__(self, start_point: np.ndarray, path_size: np.ndarray, 
        max_speed: float, r_length: float, delta: float, window: float, 
        Kv: float = 0.0, Kp: float = 0.0, car_Kv: float = 0.0, 
        car_Kp: float = 5.0) -> None:
        # path_size is the radius of two circles (in meter)
        self._circle_radius = path_size[0]
        super().__init__(start_point, path_size, max_speed, r_length, 
                        delta, window, Kv, Kp, car_Kv, car_Kp)

    def generate_path(self) -> np.ndarray:
        delta_theta = self._delta / self._circle_radius
        center_point = self._start_point
        positive_center_point = center_point + np.array([0, self._circle_radius])
        negative_center_point = center_point - np.array([0, self._circle_radius])
        point_set = []
        for i in range(int(2 * np.pi / delta_theta) + 1):
            delta_x = np.cos(- np.pi / 2 - i * delta_theta)
            delta_y = np.sin(- np.pi / 2 - i * delta_theta)
            point_set.append(positive_center_point + self._circle_radius * np.array([delta_x, delta_y]))
        for i in range(int(2 * np.pi / delta_theta) + 1):
            delta_x = np.cos(np.pi / 2 + i * delta_theta)
            delta_y = np.sin(np.pi / 2 + i * delta_theta)
            point_set.append(negative_center_point + self._circle_radius * np.array([delta_x, delta_y]))
        point_set = np.array(point_set)
        return point_set

    def get_whose_center(self) -> bool:
        # false on the negative side, true on the positive side
        if self._point_set[self._visit_position][1] >= 0:
            return True
        return False

    def get_obj_angle(self, obj_pos: np.ndarray) -> float:
        if self.get_whose_center():
            return np.arctan2(obj_pos[1] - self._circle_radius - self._start_point[1], obj_pos[0] - self._start_point[0]) - np.pi/2
        return np.arctan2(obj_pos[1] + self._circle_radius - self._start_point[1], obj_pos[0] - self._start_point[0]) + np.pi/2
    
    def get_obj_velocity(self, obj_pos: np.ndarray) -> np.ndarray:
        theta = self.get_obj_angle(obj_pos)
        if self.get_whose_center():
            return np.array([self._max_speed * np.cos(theta + np.pi/2), self._max_speed * np.sin(theta + np.pi/2)])
        return np.array([self._max_speed * np.cos(theta - np.pi/2), self._max_speed * np.sin(theta - np.pi/2)])
        
    def get_omega(self, obj_pos: np.ndarray) -> float:
        if self.get_whose_center():
            return np.linalg.norm(self.get_obj_velocity(obj_pos)) / self._circle_radius
        return - np.linalg.norm(self.get_obj_velocity(obj_pos)) / self._circle_radius

class FollowLine(FollowPath):
    def __init__(self, start_point: np.ndarray, path_size: np.ndarray, 
        max_speed: float, r_length: float, delta: float, window: float, 
        Kv: float = 0.0, Kp: float = 0.0, car_Kv: float = 0.0, 
        car_Kp: float = 5.0) -> None:
        self._path_length = path_size[0]
        super().__init__(start_point, path_size, max_speed, r_length, 
                        delta, window, Kv, Kp, car_Kv, car_Kp)

    def generate_path(self) -> np.ndarray:
        y = np.arange(self._start_point[1], self._start_point[1] + self._path_length, self._delta)
        x = self._start_point[0] * np.ones(y.shape)
        point_set = np.array([x, y]).transpose()
        return point_set
    
    def get_obj_velocity(self, obj_pos: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0])
    
    def get_omega(self, obj_pos: np.ndarray) -> float:
        return 0.0

    def get_obj_angle(self, obj_pos: np.ndarray) -> float:
        return np.pi / 2


class FollowTanh(FollowPath):
    def __init__(self, start_point: np.ndarray, path_size: np.ndarray,
        max_speed: float, r_length: float, delta: float, window: float,
        Kv: float = 0.0, Kp: float = 0.0, car_Kv: float = 0.0, 
        car_Kp: float = 5.0) -> None:
        # path_size is np.array([length, height])
        self._path_length = path_size[0]
        self._path_height = path_size[1]
        super().__init__(start_point, path_size, max_speed, r_length,
                        delta, window, Kv, Kp, car_Kv, car_Kp)
        
    def generate_path(self) -> np.ndarray:
        x0 = - self._path_length / 2
        y0 = np.tanh(2)
        A = self._path_height / (2 * np.tanh(2))
        w = 4 / self._path_length
        y = np.arange(self._start_point[1], self._start_point[1] + self._path_length, self._delta)
        x = - A * (np.tanh(w * (y - self._start_point[1] + x0)) + y0) + self._start_point[0]
        point_set = np.array([x, y]).transpose()
        return point_set
    
    def get_obj_velocity(self, obj_pos: np.ndarray) -> np.ndarray:
        # assuming velocity is not needed
        return np.array([0.0, 0.0])
    
    def get_omega(self, obj_pos: np.ndarray) -> float:
        # assuming velocity is not needed
        return 0.0
    
    def get_obj_angle(self, obj_pos: np.ndarray) -> float:
        return np.arctan2(1 - np.tanh(obj_pos[1]) ** 2, 1) + np.pi / 2

if __name__ == "__main__":
    pass