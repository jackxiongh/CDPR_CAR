import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
import numpy as np
from math import sin, cos
from stable_baselines3 import PPO, SAC
import transform
import os, sys
from follow_path import FollowLine, FollowEight, FollowTanh

class RosEnv():
    def __init__(self, topics: dict):
        # environment parameter
        self._boundary_range = 0.9       # boundary range (in object frame)
        self._max_speed = 0.2            # max speed of cars

        # thresholds
        self._reach_target_threshold = 0.07          # object reach target
        
        # init variables
        self._car_pos = np.zeros((4, 2))
        self._car_angle = np.zeros(4)
        self._object_pos = np.zeros(2)
        self._object_height = np.zeros(1)
        self._object_angle = np.zeros(1)
        self._obstacle_pos = 100.0 * np.ones(2)
        self._last_speed = np.zeros(8)
        self._target_speed = np.random.uniform(0, self._max_speed, 2)

        self._obstacle_radius = np.array([0.19])
        self._obstacle_height = np.array([1.0])

        # velocity publisher
        self._car1_vel_pub = rospy.Publisher(topics["car1_action_pub"], Twist, queue_size=2)
        self._car2_vel_pub = rospy.Publisher(topics["car2_action_pub"], Twist, queue_size=2)
        self._car3_vel_pub = rospy.Publisher(topics["car3_action_pub"], Twist, queue_size=2)
        self._car4_vel_pub = rospy.Publisher(topics["car4_action_pub"], Twist, queue_size=2)
        
        # rviz publisher
        self._car1_pos_pub = rospy.Publisher(topics["car1_pos_pub"], PoseStamped, queue_size=2)
        self._car2_pos_pub = rospy.Publisher(topics["car2_pos_pub"], PoseStamped, queue_size=2)
        self._car3_pos_pub = rospy.Publisher(topics["car3_pos_pub"], PoseStamped, queue_size=2)
        self._car4_pos_pub = rospy.Publisher(topics["car4_pos_pub"], PoseStamped, queue_size=2)
        self._obj_pos_pub = rospy.Publisher(topics["obj_pos_pub"], PoseStamped, queue_size=2)
        self._obs_pos_pub = rospy.Publisher(topics["obs_pos_pub"], PoseStamped, queue_size=2)
        self._tar_pos_pub = rospy.Publisher(topics["tar_pos_pub"], PoseStamped, queue_size=2)

        # car pos subscriber
        self._car1_pos_sub = rospy.Subscriber(topics["car1_pos_sub"], PoseStamped, callback=self.car1_callback, queue_size=2)
        self._car2_obs_sub = rospy.Subscriber(topics["car2_pos_sub"], PoseStamped, callback=self.car2_callback, queue_size=2)
        self._car3_obs_sub = rospy.Subscriber(topics["car3_pos_sub"], PoseStamped, callback=self.car3_callback, queue_size=2)
        self._car4_obs_sub = rospy.Subscriber(topics["car4_pos_sub"], PoseStamped, callback=self.car4_callback, queue_size=2)
        
        # object pos subscriber
        self._obj_pos_sub = rospy.Subscriber(topics["obj_pos_sub"], PoseStamped, callback=self.object_callback, queue_size=2)

        # obstacle pos subscriber
        self._obs_pos_sub = rospy.Subscriber(topics["obs_pos_sub"], PoseStamped, callback=self.obstacle_callback, queue_size=2)
        
        self.reset()

    def reset(self):
        self._target_pos = np.array([0.6, 1.6])
        self._target_angle = np.array([0.0])    # range([0, 360])
        self._state_norm = self.get_obs()
        # pub target
        point = PoseStamped()
        point.header.frame_id = "world"
        point.pose.position.x = self._target_pos[0]
        point.pose.position.y = self._target_pos[1]
        point.pose.position.z = 0
        w, x, y, z = transform.euler2quat(self._target_angle, 0, 0)
        point.pose.orientation.w = w
        point.pose.orientation.x = x
        point.pose.orientation.y = y
        point.pose.orientation.z = z
        self._tar_pos_pub.publish(point)

        return self._state_norm

    def get_obs(self):
        # get target, obstacle and car position in object frame
        # p^obj_i = R(-ang) * (p^world_i - p_obj^world)
        # target position in object frame
        tmp_vec = self._target_pos - self._object_pos
        self._target_pos_obj_frame = np.array(
            [np.cos(self._object_angle[0]) * tmp_vec[0] + np.sin(self._object_angle[0]) * tmp_vec[1],
            -np.sin(self._object_angle[0]) * tmp_vec[0] + np.cos(self._object_angle[0]) * tmp_vec[1]])

        # obstacle position in object frame
        tmp_vec = self._obstacle_pos - self._object_pos
        self._obstacle_pos_obj_frame = np.array(
            [np.cos(self._object_angle[0]) * tmp_vec[0] + np.sin(self._object_angle[0]) * tmp_vec[1],
            -np.sin(self._object_angle[0]) * tmp_vec[0] + np.cos(self._object_angle[0]) * tmp_vec[1]])
        
        # car position in object frame
        self._car_pos_obj_frame = []
        for i in range(4):
            tmp_vec = self._car_pos[i, :] - self._object_pos
            self._car_pos_obj_frame.append(np.cos(self._object_angle[0]) * tmp_vec[0] + np.sin(self._object_angle[0]) * tmp_vec[1])
            self._car_pos_obj_frame.append(-np.sin(self._object_angle[0]) * tmp_vec[0] + np.cos(self._object_angle[0]) * tmp_vec[1])
        self._car_pos_obj_frame = np.array(self._car_pos_obj_frame)

        # last speed in current object frame, we assume that the object angle changes very few during
        # last and this time slot
        self._last_speed_obj_frame = self._last_speed

        # normalizaton
        target_pos_norm = self._target_pos_obj_frame / self._boundary_range
        obj_height_norm = self._object_height / self._boundary_range
        obstacle_pos_norm = self._obstacle_pos_obj_frame / self._boundary_range
        obstacle_radius_norm = self._obstacle_radius / self._boundary_range
        obstacle_height_norm = self._obstacle_height / self._boundary_range
        car_pos_norm = self._car_pos_obj_frame / self._boundary_range
        last_speed_norm = self._last_speed_obj_frame / self._max_speed

        state_norm = np.concatenate((target_pos_norm,  # 2
                                obj_height_norm,       # 1
                                obstacle_pos_norm,     # 2
                                obstacle_radius_norm,  # 1
                                obstacle_height_norm,  # 1
                                car_pos_norm,          # 8
                                last_speed_norm,       # 8
                                ))
        
        # print("----------observation----------")
        # print("target pos:", self._target_pos_obj_frame)
        # print("obstacle pos:", self._obstacle_pos_obj_frame)
        # print("object radius:", self._obstacle_radius)
        # print("car pos:", self._car_pos_obj_frame)
        # print("last speed:", self._last_speed_obj_frame)
        # print("target pos norm:", target_pos_norm)
        # print("obstacle pos norm:", obstacle_pos_norm)
        # print("object radius norm:", obstacle_radius_norm)
        # print("car pos norm:", car_pos_norm)
        # print("last speed norm:", last_speed_norm)

        return state_norm


    def is_reach_target(self):
        if (np.linalg.norm(self._target_pos_obj_frame) <= self._reach_target_threshold):
            return True
        return False

    def car1_callback(self, msg: PoseStamped):
        self._car_pos[0, :] = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        y = transform.quat2euler(msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w)
        self._car_angle[0] = y
        self._car_angle[0] = self.angle_limit(self._car_angle[0])
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._car1_pos_pub.publish(msg)

    def car2_callback(self, msg: PoseStamped):
        self._car_pos[1, :] = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        y = transform.quat2euler(msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w)
        self._car_angle[1] = y
        self._car_angle[1] = self.angle_limit(self._car_angle[1])
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._car2_pos_pub.publish(msg)

    def car3_callback(self, msg: PoseStamped):
        self._car_pos[2, :] = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        y = transform.quat2euler(msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w)
        self._car_angle[2] = y
        self._car_angle[2] = self.angle_limit(self._car_angle[2])
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._car3_pos_pub.publish(msg)

    def car4_callback(self, msg: PoseStamped):
        self._car_pos[3, :] = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        y = transform.quat2euler(msg.pose.orientation.x, msg.pose.orientation.y,
                                        msg.pose.orientation.z, msg.pose.orientation.w)
        self._car_angle[3] = y
        self._car_angle[3] = self.angle_limit(self._car_angle[3])
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._car4_pos_pub.publish(msg)

    def object_callback(self, msg: PoseStamped):
        self._object_pos = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        self._object_height = np.array([msg.pose.position.z / 1000.0])
        y = transform.quat2euler(msg.pose.orientation.x, msg.pose.orientation.y,
                                    msg.pose.orientation.z, msg.pose.orientation.w)
        self._object_angle = np.array([y])
        self._object_angle = self.angle_limit(self._object_angle)
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._obj_pos_pub.publish(msg)

    def obstacle_callback(self, msg: PoseStamped):
        self._obstacle_pos = np.array([msg.pose.position.x / 1000.0, msg.pose.position.y / 1000.0])
        msg.pose.position.x = msg.pose.position.x / 1000.0
        msg.pose.position.y = msg.pose.position.y / 1000.0
        msg.pose.position.z = msg.pose.position.z / 1000.0
        self._obs_pos_pub.publish(msg)

    def angle_limit(self, ang):
        while (ang > 2 * np.pi) or (ang < -2 * np.pi):
            if ang > 2 * np.pi:
                ang = ang - 2 * np.pi
            if ang < -2 * np.pi:
                ang = ang + 2 * np.pi
        return ang

    def in_range(self):
        if np.linalg.norm(self._obstacle_pos_obj_frame) < self._boundary_range:
            return True
        return False

    def step(self, action) -> np.ndarray:
        self.step_no_obs(action)
        self._state_norm = self.get_obs()
        return self._state_norm
    
    def step_no_obs(self, action) -> None:
        speed = action * self._max_speed
        twist_pub = Twist()
        twist_pub.linear.x = speed[0]
        twist_pub.linear.y = speed[1]
        self._car1_vel_pub.publish(twist_pub)
        twist_pub.linear.x = speed[2]
        twist_pub.linear.y = speed[3]
        self._car2_vel_pub.publish(twist_pub)
        twist_pub.linear.x = speed[4]
        twist_pub.linear.y = speed[5]
        self._car3_vel_pub.publish(twist_pub)
        twist_pub.linear.x = speed[6]
        twist_pub.linear.y = speed[7]
        self._car4_vel_pub.publish(twist_pub)

        self._last_speed = speed
    
    def get_car_pos(self) -> np.ndarray:
        return self._car_pos

    def get_obj_angle(self) -> float:
        return self._object_angle[0]
    
    def get_obj_pos(self) -> np.ndarray:
        return self._object_pos
    
    def get_car_angle(self) -> np.ndarray:
        return self._car_angle
    
    def set_target_pos(self, target_pos):
        self._target_pos = target_pos
    
    def get_obstacle_pos(self) -> np.ndarray:
        return self._obstacle_pos
    
    def get_obs_obj_dis(self) -> float:
        return np.linalg.norm(self._obstacle_pos_obj_frame)
    
    def get_obstacle_radius(self) -> np.ndarray:
        return self._obstacle_radius
    
    def action_obj_2_car(self, action_obj_frame) -> np.ndarray:
        action_car_frame = np.zeros(8)
        obj_angle = self.get_obj_angle()
        car_angle = self.get_car_angle()
        ang = car_angle - obj_angle
        for i in range(4):
            action_car_frame[2*i:2*i+2] = np.array([np.cos(ang[i]) * action_obj_frame[2*i] + np.sin(ang[i]) * action_obj_frame[2*i+1], 
                                        - np.sin(ang[i]) * action_obj_frame[2*i] + np.cos(ang[i]) * action_obj_frame[2*i+1]])
        return action_car_frame

class RosWrapper():
    def __init__(self, node: str, topics: list, model: SAC, move_shape: int, if_avoid: bool):
        """
        move_shape: 1 for moving in line, 2 for moving in 8-shape, 3 for moving in tanh
        """
        rospy.init_node(node)
        self._shape = move_shape
        self._env = RosEnv(topics)
        self._model = model
        self._obs = self._env.reset()
        self._if_avoid = if_avoid
        self._set_target = False
        self._cnt = 0

        # set obstacle params
        if self._shape == 1:
            self._obstacle_pos = np.array([[0.0, 0.0]])
            self._obstacle_radius = np.array([[0.175]])
        elif self._shape == 2:
            self._obstacle_pos = np.array([[-1.6, 1.6], [1.6, -1.6]])
            self._obstacle_radius = np.array([[0.175], [0.16]])
        elif self._shape == 3:
            self._obstacle_pos = np.array([[0.0, 0.0]])
            self._obstacle_radius = np.array([[1.0]])

        # set moving path
        if move_shape == 1:
            start_point = np.array([0.0, 0.55])
            path_size = np.array([3.3])
            self._follow_path = FollowLine(start_point, path_size, 1.0, 
                                            0.4, 0.15, 0.8, 0.0, 0.0, 0.0, 4.5)
        elif move_shape == 2:
            start_point = np.array([0.0, 0.0])
            path_size = np.array([1.6])
            self._follow_path = FollowEight(start_point, path_size, 1.0, 
                                            0.55, 0.07, 0.8)
        elif move_shape == 3:
            start_point = np.array([1.15, 0.2])
            path_size = np.array([3.6, 2.0])
            self._follow_path = FollowTanh(start_point, path_size, 1.0,
                                            0.4, 0.15, 0.8, 0.0, 0.0, 0.0, 5)

        rospy.Timer(rospy.Duration(1.0 / 50.0), self.timer_callback)
        rospy.spin()
    
    def mixer(self, distance, radius, v_ff, v_rl) -> np.ndarray:
        x = (distance - radius - 0.85) / 0.15
        alpha = 1 / (1 + np.exp(-x))
        vel = alpha * v_ff + (1 - alpha) * v_rl
        return vel

    def timer_callback(self, event):
        # if need to avoid obstacle
        if self._if_avoid and self._env.in_range():
            self._cnt = 0
            if not self._set_target:
                # set the target for set_obstacle^th obstacle
                self._env.set_target_pos(self._follow_path.get_target_point(
                    self._env.get_obstacle_pos(), self._env.get_obstacle_radius() + 0.90)[0])
                self._env._last_speed = np.zeros(8)
                self._set_target = False
            # keep the path point updated
            action_ff, car_tar_pos = self._follow_path.get_car_velocity(self._env.get_car_pos(), 
                self._env.get_obj_pos(), self._env.get_obj_angle(), 
                self._env.get_car_angle() - self._env.get_obj_angle())
            self._obs = self._env.get_obs()
            # real action
            action_rl, _state = self._model.predict(self._obs)
            action_rl = self._env.action_obj_2_car(action_rl)
            action = self.mixer(self._env.get_obs_obj_dis(), self._env._obstacle_radius, action_ff, action_rl)
        # if just need to go along the path
        else:
            # We split the observation and the action in real world, 
            # in order to make the time slot between the action and 
            # the observation the same as that of the training environment.
            if self._cnt == 5:
                self._set_target = False
                self._cnt += 1
            elif self._cnt < 5:
                self._cnt += 1
            action, _ = self._follow_path.get_car_velocity(self._env.get_car_pos(), 
                self._env.get_obj_pos(), self._env.get_obj_angle(), 
                self._env.get_car_angle() - self._env.get_obj_angle())
                        
        self._env.get_obs()
        self._env.step_no_obs(action)