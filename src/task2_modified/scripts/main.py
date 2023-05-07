import logging
import time
from mujoco_train import Omni_drag_object_env as env
from mujoco_test import TestEnv
import argparse
import torch as th
import os, sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from eval_callback import EvalCallback, evaluate
from ros_env import RosWrapper
from ros_topic_name import topics_name
from attention_sac import SACAttentionPolicy

logger = logging.getLogger(__name__)
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--env_num", type = int, default = 12, help="同步训练的环境个数")
ap.add_argument("-t", "--train_timestep", type = int, default = 80000000, help="训练的次数") # 80000000
ap.add_argument("-m", "--type", type = str, default = 'ros', help="'train', 'eval', 'continue_train', 'ros', 'test'")
ap.add_argument("-d", "--device", type = str,  default='cuda', help="cpu / cuda")
ap.add_argument("-k", "--keep_train_model_path", type = str,  default=os.path.abspath(os.path.join(sys.path[0], "../model/best_model")), help="继续训练的模型路径")
ap.add_argument("-e", "--eval_model_path", type = str,  default=os.path.abspath(os.path.join(sys.path[0], "../model/best_model")), help="用于评估的模型路径")
ap.add_argument("-p", "--train_path", type=str, default=os.path.abspath(os.path.join(sys.path[0], "../model_log/model")))
args = vars(ap.parse_args())

if __name__ == '__main__':
    method = str(args["type"])

    on_policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])])
    
    off_policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=dict(pi=[128, 256, 256], qf=[128, 256, 256])) # (64, 64), (64, 64); (64, 64, 64), (64, 64, 64) 

    if method == "train":      # train
        checkpoint_callback = EvalCallback(save_freq=5000, save_path=args["train_path"])
        env_train = SubprocVecEnv([lambda: Monitor(env(False)) for _ in range(int(args["env_num"]))])
        # model = SAC("MlpPolicy", env_train, verbose=1, policy_kwargs=off_policy_kwargs, device=args["device"], tensorboard_log="./tensorboard")
        model = SAC(SACAttentionPolicy, env_train, verbose=1, policy_kwargs=off_policy_kwargs, device=args["device"], tensorboard_log="./tensorboard")
        print(model.policy)
        model.learn(total_timesteps = args["train_timestep"], callback=checkpoint_callback)
        model.save(args["train_path"] + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    elif method == "eval":      # eval
        env_test = env(render = True)
        model = SAC.load(args["eval_model_path"], env_test)
        evaluate(model, env_test, num_episodes=1000)
    elif method == "continue_train":      # continue train
        checkpoint_callback = EvalCallback(save_freq=5000, save_path=args["train_path"])
        env_train = SubprocVecEnv([lambda: Monitor(env(False)) for i in range(args["env_num"])])
        model = SAC.load(args["keep_train_model_path"], env_train)
        model.learn(total_timesteps = args["train_timestep"], callback=checkpoint_callback)
        model.save(args["train_path"] + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    elif method == "ros":
        node_name = 'ros_env'
        model = SAC.load(args["eval_model_path"])
        ros_wrapper = RosWrapper(node_name, topics_name, model, 3, True)
    elif method == "test":
        model = SAC.load(args["eval_model_path"])
        test_env = TestEnv(model, 3, True)
        test_env.test()
