from cv2 import getTickFrequency
import gym
import numpy as np
from mujoco_controller import car_mujoco_env

def normalize_ang(ang1):
    while ang1 > 180:
        ang1 -= 360
    while ang1 < -180:
        ang1 += 360
    return ang1

class Omni_drag_object_env(gym.Env):
    def __init__(self, render=False, train=True):
        # env
        self.env = car_mujoco_env(render=render, train=train)
        self.render = render

        # environment parameter
        self.boundary_range = 0.9       # boundary range (in object frame)
        self.field_range = 1.5          # half of the field size
        self.max_step = 30              # max simulation step, max time = max_step * 50(real simulation step) * 0.01(RK time slice)
        self.target_span = 0.8          # max distance from target to boundary
        self.max_speed = 0.2            # max speed of cars
        self.obstacle_max_radius = 0.2  # max obstacle radius
        self.obstacle_max_height = 0.32  # max obstacle height

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
        self.car_obs_near_thresh = 0.2
        self.car_obs_near_slope = 0 # 500

        # avoid car-car collision reward function
        # _/\_
        self.car_car_ac_thresh = 0.38
        self.car_car_ac_slope = 200 # 200

        # avoid car-object collision reward function
        # \    /  \    /
        #  \__/    \__/
        self.car_obj_far_slope = 20 # 20
        self.car_obj_far_thresh = 0.5
        self.car_obj_near_thresh = 0.2
        self.car_obj_near_slope = 20 # 20

        # avoid object-obstacle collision reward function (x, y)
        #     ____
        #    /    \
        # __/      \__
        self.obj_obs_xy_thresh = 0.0

        # avoid object-obstacle collision reward function (z)
        # \
        #  \
        #   \____
        self.obj_obs_z_thresh = 0.2
        self.obj_obs_slope = 150 # 50
        
        # thresholds
        self.reach_target_threshold = 0.07          # object reach target

        # factors
        self.distance_factor = 10           # object reach target
        self.object_steady_factor = 2       # object steady

        # settings
        self.boundary = [-self.boundary_range, self.boundary_range]
        self.observation_space = gym.spaces.Box(low = -1, high = 1 ,shape=(23,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low = -1, high = 1 ,shape=(8,), dtype=np.float64)

        # reset
        self.reset()
    
    def reset(self):
        # generate obstacle position
        obstacle_pos_radius = np.random.uniform(self.boundary_range / 1.5, self.boundary_range, 1)
        obstacle_pos_angle = np.random.uniform(0, 2 * np.pi, 1)
        self.obstacle_pos = np.array(
            [np.cos(obstacle_pos_angle[0]) * obstacle_pos_radius[0], np.sin(obstacle_pos_angle[0]) * obstacle_pos_radius[0]])
        self.obstacle_radius = np.random.uniform(self.obstacle_max_radius * 0.75, self.obstacle_max_radius, 1)
        self.obstacle_height = np.random.uniform(0.2, self.obstacle_max_height, 1)
        
        # generate target position
        target_pos_radius = np.random.uniform(
            self.obstacle_radius + self.boundary_range / 1.5, self.obstacle_radius + self.boundary_range, 1)
        target_pos_angle = np.random.uniform(-np.pi/3, np.pi/3, 1)
        target_pos_angle = obstacle_pos_angle + target_pos_angle
        self.target_pos = self.obstacle_pos + np.array(
            [np.cos(target_pos_angle[0]) * target_pos_radius[0], np.sin(target_pos_angle[0]) * target_pos_radius[0]])

        # last speed
        self.last_speed = np.zeros(8)

        self.env.reset()

        self.env.set_target_ref_pos(self.target_pos)
        self.env.set_obstacle_pos(self.obstacle_pos, self.obstacle_radius, self.obstacle_height/2)

        self.step_count = 0
        self.state_norm = self.get_obs()
        self.if_reach_target = False
        
        return self.state_norm
        
    def get_obs(self) -> np.ndarray:
        # get car and object position and orientation from mujoco
        self.car_pos = self.env.get_car_pos()
        self.object_pos = self.env.get_target_object_pos()
        self.object_height = self.env.get_target_object_height()
        self.object_angle = self.env.get_target_object_roll_angle()
        obj_ang = self.object_angle / 180 * np.pi
        
        # test delete
        # self.obstacle_pos = self.env.get_obstacle_pos()
        # self.obstacle_radius = np.array([self.env.get_obstacle_radius()])

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
        

        if self.render:
            print("----------observation", self.step_count, "----------")
            print("target pos:", self.target_pos_obj_frame)
            print("obstacle pos:", self.obstacle_pos_obj_frame)
            print("object radius:", self.obstacle_radius)
            print("object height:", self.obstacle_height)
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
    
    # get_obs must be called before get_reward
    def get_reward(self) -> np.ndarray:
        reward = 0

        # avoid car-car collision
        avoid_car_car_collision_reward = self.get_car_car_distance_reward()

        # avoid car-obstacle collision
        avoid_car_obstacle_collision_reward = self.get_car_obstacle_distance_reward()
        
        # avoid car-object too near or too far
        avoid_car_object_collision_reward = self.get_car_object_distance_reward()

        # avoid object-obstacle collision
        avoid_object_obstacle_collision_reward = self.get_object_obstacle_distance_reward()
                
        # position reach target
        position_reward = - self.distance_factor * np.linalg.norm(self.target_pos_obj_frame)

        # keep object steady
        object_steady_reward = - self.object_steady_factor * \
                np.abs((normalize_ang(self.object_angle[1]) + normalize_ang(self.object_angle[2]))) / 2 / 180
        
        # reach target bonus
        reach_target_reward = 0
        if self.is_reach_target():
            reach_target_reward += 0 # 2000

        # total reward
        reward = avoid_car_car_collision_reward + avoid_car_obstacle_collision_reward + \
                avoid_car_object_collision_reward + avoid_object_obstacle_collision_reward + \
                position_reward + reach_target_reward

        if self.render:
            print("----------reward", self.step_count, "----------")
            print("car-car collision:", avoid_car_car_collision_reward)
            print("car-obstacle collision:", avoid_car_obstacle_collision_reward)
            print("car-object collision:", avoid_car_object_collision_reward)
            print("position:", position_reward)
            print("object steady:", object_steady_reward)

        return reward

    def step(self, action):
        speed = action * self.max_speed
        self.last_speed = speed

        # In mujoco, the car can not rotate, so the angle 
        # between the object and the car is the angle of the object
        ang = self.env.get_car_angle() - self.object_angle[0] / 180.0 * np.pi
        for i in range(4):
            speed[2*i:2*i+2] = np.array([np.cos(ang[i]) * speed[2*i] + np.sin(ang[i]) * speed[2*i+1],
            - np.sin(ang[i]) * speed[2*i] + np.cos(ang[i]) * speed[2*i+1]])

        flag = self.env.control(speed)

        # handle error 
        if flag == False:
            return self.state_norm, np.array(-1000), True, {}
        
        # get_obs must be called before get_reward
        self.state_norm = self.get_obs()
        reward = self.get_reward()
        end_flag = self.get_end_flag()
        self.step_count = self.step_count + 1

        return self.state_norm, reward, end_flag, {}

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
                    if self.render:
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
            if self.render:
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
    
    def get_object_obstacle_distance_reward(self):
        avoid_object_obstacle_collision_reward = 0
        obj_obs_xy_dis = np.linalg.norm(self.obstacle_pos_obj_frame) - \
                    self.obstacle_radius[0]
        obj_obs_z_dis = self.object_height[0] - self.obstacle_height[0]
        if obj_obs_xy_dis < self.obj_obs_xy_thresh and \
            obj_obs_z_dis < self.obj_obs_z_thresh :
            # use hieght constraint
            if obj_obs_xy_dis < 0:
                avoid_object_obstacle_collision_reward -= self.obj_obs_slope * \
                    (self.obj_obs_z_thresh - obj_obs_z_dis)
            elif obj_obs_z_dis < 0:
                avoid_object_obstacle_collision_reward -= self.obj_obs_slope * \
                    (self.obj_obs_xy_thresh - obj_obs_xy_dis)
        return avoid_object_obstacle_collision_reward

    # judge if done
    def get_end_flag(self):
        # reach target or exceed max step
        if self.is_reach_target() or self.is_exceed_max_step():
            self.if_reach_target = True
            return True
        return False
    
    def is_exceed_max_step(self):
        if self.step_count >= self.max_step:
            return True
        return False
    
    def is_reach_target(self):
        if (np.linalg.norm(self.target_pos_obj_frame) <= self.reach_target_threshold):
            return True
        return False


def test():
    env = Omni_drag_object_env(True)
    while True:
        env.reset()
        while True:
            action = env.action_space.sample()
            env.step(action)
            if env.step_count > 3:
                break


if __name__ == '__main__':
    obj = Omni_drag_object_env()
    h = obj.env.get_target_object_height()
    print(h)
