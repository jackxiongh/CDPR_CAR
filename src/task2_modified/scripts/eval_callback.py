from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, SAC
from mujoco_train import Omni_drag_object_env as env
import numpy as np
import os

def evaluate(model, env, num_episodes=100):
    # This function will only work for a single Environment
    all_episode_rewards = []
    success_num = 0
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            print("------------action------------")
            print(action)
            obs, reward, done, info = env.step(action)
            if_reach = env.if_reach_target
            episode_rewards.append(reward)
            if if_reach:
                success_num +=1
        print("\r" + str(100 * i / num_episodes) + "% trial completed", end='', flush=True, sep=None)
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("平均回报：{} 成功率：{}".format(mean_episode_reward, 100 * success_num/num_episodes))
    return mean_episode_reward,success_num/num_episodes

class EvalCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(EvalCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._ros_env = env(render = True)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            model = SAC.load(path, self._ros_env)
            evaluate(model, self._ros_env, num_episodes=1)
        return True