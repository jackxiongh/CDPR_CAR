import gym
import numpy as np
from mujoco_controller import car_mujoco_env
from follow_path import FollowLine, FollowEight, FollowTanh
from stable_baselines3 import SAC

def normalize_ang(ang1):
    while ang1 > 180:
        ang1 -= 360
    while ang1 < -180:
        ang1 += 360
    return ang1

class Omni_drag_object_env(gym.Env):
    def __init__(self, render=False, mode=1, output=False):
        # env
        self.env = car_mujoco_env(render=render, train=False, random_move=False, mode=mode)
        self.render = render
        self.output = output

        # environment parameter
        self.boundary_range = 0.9       # boundary range (in object frame)
        self.field_range = 1.5          # half of the field size
        self.max_step = 30              # max simulation step, max time = max_step * 50(real simulation step) * 0.01(RK time slice)
        self.target_span = 0.8          # max distance from target to boundary
        self.max_speed = 0.2            # max speed of cars
        self.obstacle_max_radius = 0.3  # max obstacle radius 

        # reward functions parameters
        # avoid car-obstacle collision reward function
        # \    /  \    /
        #  \__/    \__/
        # outer_slope(\) outer_threshold(\_) inner_threshold(_/) inner_slope(/)
        # inner and outer threshold plus the radius is the distance from the center
        # \   end   \
        #  \         \
        #   \         \
        #   |  (obs)  |
        #   |         |
        #   |  start  |
        self.car_obs_far_slope = 50     # not used
        self.car_obs_far_thresh = 0.3   # not used
        self.car_obs_near_thresh = 0.2
        self.car_obs_near_slope = 500

        # avoid car-car collision reward function
        # _/\_
        self.car_car_ac_thresh = 0.38
        self.car_car_ac_slope = 200

        # avoid car-object collision reward function
        # \    /  \    /
        #  \__/    \__/
        self.car_obj_far_slope = 20
        self.car_obj_far_thresh = 0.6
        self.car_obj_near_thresh = 0.2
        self.car_obj_near_slope = 20

        # avoid car deformation reward function
        self.car_deformation_range = 20             # half the deformation range
        
        # thresholds
        self.reach_target_threshold = 0.07          # object reach target

        # factors
        self.distance_factor = 10           # object reach target
        self.object_steady_factor = 2       # object steady

        # reset
        self.reset()
    
    def reset(self):
        # target pos
        self.target_pos = np.array([6.0, 6.0])

        # obstacle pos
        self.obstacle_pos = np.array([6.0, 6.0])
        self.obstacle_radius = np.array([0.5])
        self.obstacle_height = np.array([0.0])

        # last speed
        self.last_speed = np.zeros(8)

        self.env.reset()

        self.step_count = 0
        self.state_norm = self.get_obs()
        self.if_reach_target = False
        
        return self.state_norm
        
    def get_obs(self):
        # get car and object position and orientation from mujoco
        self.car_pos = self.env.get_car_pos()
        self.object_pos = self.env.get_target_object_pos()
        self.object_height = self.env.get_target_object_height()
        self.object_angle = self.env.get_target_object_roll_angle()
        obj_ang = self.object_angle / 180 * np.pi

        # get target, obstacle and car position in object frame
        # p^obj_i = R(-ang) * (p^world_i - p_obj^world)
        # target position in object frame
        tmp_vec = self.target_pos - self.object_pos
        self.target_pos_obj_frame = np.array(
            [np.cos(obj_ang[0]) * tmp_vec[0] + np.sin(obj_ang[0]) * tmp_vec[1],
            -np.sin(obj_ang[0]) * tmp_vec[0] + np.cos(obj_ang[0]) * tmp_vec[1]])

        # obstacle position in object frame
        tmp_vec = self.obstacle_pos - self.object_pos
        self.obstacle_pos_obj_frame = np.array(
            [np.cos(obj_ang[0]) * tmp_vec[0] + np.sin(obj_ang[0]) * tmp_vec[1],
            -np.sin(obj_ang[0]) * tmp_vec[0] + np.cos(obj_ang[0]) * tmp_vec[1]])

        # car position in object frame
        self.car_pos_obj_frame = []
        for i in range(4):
            tmp_vec = self.car_pos[i, :] - self.object_pos
            self.car_pos_obj_frame.append(np.cos(obj_ang[0]) * tmp_vec[0] + np.sin(obj_ang[0]) * tmp_vec[1])
            self.car_pos_obj_frame.append(-np.sin(obj_ang[0]) * tmp_vec[0] + np.cos(obj_ang[0]) * tmp_vec[1])
        self.car_pos_obj_frame = np.array(self.car_pos_obj_frame)

        # last speed in current object frame, we assume that the object angle changes very few during
        # last and this time slot
        self.last_speed_obj_frame = self.last_speed

        # normalizaton
        target_pos_norm = self.target_pos_obj_frame / self.boundary_range
        obj_height_norm = self.object_height / self.boundary_range
        obstacle_pos_norm = self.obstacle_pos_obj_frame / self.boundary_range
        obstacle_radius_norm = self.obstacle_radius / self.boundary_range
        obstacle_height_norm = self.obstacle_height / self.boundary_range
        car_pos_norm = self.car_pos_obj_frame / self.boundary_range
        last_speed_norm = self.last_speed_obj_frame / self.max_speed

        if self.output:
            print("----------observation", self.step_count, "----------")
            print("object ang:", obj_ang[0])
            print("target pos:", self.target_pos_obj_frame)
            print("obstacle pos:", self.obstacle_pos_obj_frame)
            print("object radius:", self.obstacle_radius)
            print("car pos:", self.car_pos_obj_frame)
            print("last speed:", self.last_speed_obj_frame)

        state_norm = np.concatenate((target_pos_norm,  # 2
                                obj_height_norm,       # 1
                                obstacle_pos_norm,     # 2
                                obstacle_radius_norm,  # 1
                                obstacle_height_norm,  # 1
                                car_pos_norm,          # 8
                                last_speed_norm,       # 8
                                ))
        return state_norm
    
    def get_reward(self):
        reward = 0

        # avoid car-car collision
        avoid_car_car_collision_reward = self.get_car_car_distance_reward()

        # avoid car-obstacle collision
        avoid_car_obstacle_collision_reward = self.get_car_obstacle_distance_reward()
        
        # avoid car-object too near or too far
        avoid_car_object_collision_reward = self.get_car_object_distance_reward()
                
        # position reach target
        position_reward = - self.distance_factor * np.linalg.norm(self.target_pos_obj_frame)

        # keep object steady
        object_steady_reward = - self.object_steady_factor * \
                np.abs((normalize_ang(self.object_angle[1]) + normalize_ang(self.object_angle[2]))) / 2 / 180

        # total reward
        reward = avoid_car_car_collision_reward + avoid_car_obstacle_collision_reward + \
                avoid_car_object_collision_reward + \
                position_reward

        if self.output:
            print("----------reward", self.step_count, "----------")
            print("car-car collision:", avoid_car_car_collision_reward)
            print("car-obstacle collision:", avoid_car_obstacle_collision_reward)
            print("car-object collision:", avoid_car_object_collision_reward)
            print("position:", position_reward)
            print("object steady:", object_steady_reward)

        return reward

    def step(self, action):
        speed = action * self.max_speed

        flag = self.env.control(speed)

        # handle error 
        if flag == False:
            return self.state_norm, -1000, True,{}
        
        # get_obs must be called before get_reward
        self.state_norm = self.get_obs()
        self.last_speed = speed

        return self.state_norm
    
    def get_car_car_distance_reward(self):
        avoid_car_car_collision_reward = 0
        for i in range(4):
            for j in range(4):
                if not i == j:
                    car_car_distance = \
                        np.linalg.norm(self.car_pos_obj_frame[2*i:2*i+2] - self.car_pos_obj_frame[2*j:2*j+2])
                    if  car_car_distance < self.car_car_ac_thresh:
                        avoid_car_car_collision_reward -= self.car_car_ac_slope * \
                            (self.car_car_ac_thresh - car_car_distance)
                    if self.output:
                        print("car", i, "car", j, "dis:", car_car_distance)
        return avoid_car_car_collision_reward


    def get_car_obstacle_distance_reward(self):
        avoid_car_obstacle_collision_reward = 0
        for i in range(4):
            # the distance between the car center and the obestacle boundary
            car_obs_boundary_distance = \
                np.linalg.norm(self.car_pos_obj_frame[2*i:2*i+2] - self.obstacle_pos_obj_frame) \
                - self.obstacle_radius[0]
            # details of the function are in __init__
            # avoid collision
            if car_obs_boundary_distance < self.car_obs_near_thresh:
                avoid_car_obstacle_collision_reward -= self.car_obs_near_slope * \
                    (self.car_obs_near_thresh - car_obs_boundary_distance)
            if self.output:
                print("car", i+1, "obs dis:", car_obs_boundary_distance)
        return avoid_car_obstacle_collision_reward
    
    def get_car_object_distance_reward(self):
        avoid_car_object_collision_reward = 0
        for i in range(4):
            car_obj_distance = np.linalg.norm(self.car_pos_obj_frame[2*i:2*i+2])
            # avoid collision
            if car_obj_distance < self.car_obj_near_thresh:
                avoid_car_object_collision_reward -= self.car_obj_near_slope * \
                    (self.car_obj_near_thresh - car_obj_distance)
            # avoid high cable tension
            elif car_obj_distance > self.car_obj_far_thresh:
                avoid_car_object_collision_reward -= self.car_obj_far_slope * \
                    (car_obj_distance - self.car_obj_far_thresh)
        return avoid_car_object_collision_reward
    
    def in_range(self):
        if np.linalg.norm(self.obstacle_pos_obj_frame) < self.boundary_range:
            return True
        return False
    
    def get_car_pos(self) -> np.ndarray:
        return self.car_pos
    
    def get_obj_pos(self) -> np.ndarray:
        return self.object_pos
    
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos
        self.env.set_target_ref_pos(self.target_pos)
    
    def get_obstacle_pos(self) -> np.ndarray:
        return self.obstacle_pos

    def set_obstacle_pos(self, obstacle_pos, obstacle_radius, obstacle_height) -> np.ndarray:
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        self.obstacle_height = obstacle_height
        self.env.set_obstacle_pos(self.obstacle_pos, self.obstacle_radius, obstacle_height/2)
    
    def get_obstacle_radius(self) -> np.ndarray:
        return self.obstacle_radius
    
    def get_obj_angle(self) -> float:
        return self.object_angle[0] / 180 * np.pi
    
    def get_car_angle(self) -> np.ndarray:
        return self.env.get_car_angle()
    
    def get_obs_obj_dis(self) -> float:
        return np.linalg.norm(self.obstacle_pos_obj_frame)
    
    def action_obj_2_car(self, action_obj_frame) -> np.ndarray:
        action_car_frame = np.zeros(8)
        obj_angle = self.get_obj_angle()
        car_angle = self.get_car_angle()
        ang = car_angle - obj_angle
        for i in range(4):
            action_car_frame[2*i:2*i+2] = np.array([np.cos(ang[i]) * action_obj_frame[2*i] + np.sin(ang[i]) * action_obj_frame[2*i+1], 
                                        - np.sin(ang[i]) * action_obj_frame[2*i] + np.cos(ang[i]) * action_obj_frame[2*i+1]])
        return action_car_frame

class TestEnv(object):
    def __init__(self, model: SAC, move_shape: int, if_avoid: bool) -> None:
        self._shape = move_shape
        self._env = Omni_drag_object_env(render=True, mode=move_shape)
        self._model = model
        self._obs = self._env.reset()
        self._if_avoid = if_avoid
        
        # set obstacle params
        if self._shape == 1:
            self._obstacle_pos = np.array([[0.0, 0.0]])
            self._obstacle_radius = np.array([[0.175]])
            self._obstacle_height = np.array([[1.0]])
        elif self._shape == 2:
            self._obstacle_pos = np.array([[-1.6, 1.6], [1.6, -1.6]])
            self._obstacle_radius = np.array([[0.175], [0.16]])
            self._obstacle_height = np.array([[0.32], [1.0]])
        elif self._shape == 3:
            self._obstacle_pos = np.array([[0.12, 0.12]])
            self._obstacle_radius = np.array([[0.175]])
            self._obstacle_height = np.array([[1.0]])

        # set moving path
        if move_shape == 1:
            start_point = np.array([0.0, -2.0])
            path_size = np.array([4])
            self._follow_path = FollowLine(start_point, path_size, 1.0, 
                                            0.55, 0.07, 0.8)
        elif move_shape == 2:
            start_point = np.array([0.0, 0.0])
            path_size = np.array([1.6])
            self._follow_path = FollowEight(start_point, path_size, 1.0, 
                                            0.55, 0.07, 0.8)
        elif move_shape == 3:
            start_point = np.array([1.0, -2.0])
            path_size = np.array([4.0, 2.0])
            self._follow_path = FollowTanh(start_point, path_size, 1.0,
                                            0.55, 0.07, 0.8)
    
    def mixer(self, distance, radius, v_ff, v_rl) -> np.ndarray:
        x = (distance - radius - 0.7) / 1e-5
        alpha = 1 / (1 + np.exp(-x))
        vel = alpha * v_ff + (1 - alpha) * v_rl
        return vel

    def test(self):
        cnt = 0
        set_obstacle = 0
        # in case the obs is wrong
        action = np.zeros(8)
        self._env.step(action)

        while True:
            # if need to avoid obstacle
            if self._if_avoid and self._env.in_range():
                cnt = 0
                self._env.set_target_pos(self._follow_path.get_target_point(
                    self._env.get_obstacle_pos(), self._env.get_obstacle_radius() + 0.7)[0])
                # keep the path point updated
                action_ff, car_tar_pos = self._follow_path.get_car_velocity(self._env.get_car_pos(), 
                    self._env.get_obj_pos(), self._env.get_obj_angle(), 
                    self._env.get_car_angle() - self._env.get_obj_angle())
                action_rl, _state = self._model.predict(self._obs)
                action_rl = self._env.action_obj_2_car(action_rl)
                # action = action_rl
                action = self.mixer(self._env.get_obs_obj_dis(), self._obstacle_radius[set_obstacle-1], action_ff, action_rl)
                print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t" % 
                    (self._env.get_obs_obj_dis(), 
                    action_ff[0], action_ff[1], action_ff[2], action_ff[3],
                    action_ff[4], action_ff[5], action_ff[6], action_ff[7],
                    action_rl[0], action_rl[1], action_rl[2], action_rl[3],
                    action_rl[4], action_rl[5], action_rl[6], action_rl[7]))
            # if just need to go along the path
            else:
                # To judge if the car is away from the obstacle, if so, set the next obstacle
                if cnt == 5:
                    # the car is away from the obstacle
                    if set_obstacle < self._obstacle_pos.shape[0]:
                        self._env.set_obstacle_pos(self._obstacle_pos[set_obstacle], self._obstacle_radius[set_obstacle], self._obstacle_height[set_obstacle])
                    set_obstacle += 1
                    cnt += 1
                elif cnt < 5:
                    cnt += 1

                action, car_tar_pos = self._follow_path.get_car_velocity(self._env.get_car_pos(), 
                    self._env.get_obj_pos(), self._env.get_obj_angle(), 
                    self._env.get_car_angle() - self._env.get_obj_angle())

            self._env.env.set_car_target_pos(car_tar_pos)
            self._obs = self._env.step(action)