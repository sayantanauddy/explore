import os
import time
import hydra
import h5py
import imageio
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from omegaconf import DictConfig, OmegaConf

from explore.utils.vis import play_path
from explore.utils.logger import get_logger
from stable_baselines3.common.env_util import make_vec_env
from explore.env.fingerRamp import FingerRampHerEnv


@hydra.main(version_base="1.3", config_path="../configs", config_name="HER")
def main(cfg: DictConfig):

    model_path = "/home/sayantan/repos/tub/proj/explore/scripts/outputs/2025-11-11/14-33-35/models/checkpoint_1960000_steps.zip"
    n_eval_episodes = 1

    s_cfg_idx = 0
    e_cfg_idx = 25

    def make_env():
        env = FingerRampHerEnv(cfg)
        env.start_config_idx = s_cfg_idx
        env.end_config_idx = e_cfg_idx
        return env

    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env.render_mode = "rgb_array"
    
    model = SAC.load(model_path, env=eval_env, device=cfg.device)

    for i in range(n_eval_episodes):
        
        obs, init_info = eval_env.envs[0].reset()

        for k,v in obs.items():
            print(k, v.shape)

        stable_configs = h5py.File(cfg.env.stable_configs_path, 'r')

        # Set eval_env to init_state and view
        eval_env.envs[0].env.sim.pushConfig(
            stable_configs["qpos"][s_cfg_idx],
            stable_configs["ctrl"][s_cfg_idx]
        )
        frame_s = eval_env.render("rgb_array")

        # Set eval_env to target_state and view
        eval_env.envs[0].env.sim.pushConfig(
            stable_configs["qpos"][e_cfg_idx],
            stable_configs["ctrl"][e_cfg_idx]
        )
        frame_e = eval_env.render("rgb_array")

        obs, init_info = eval_env.envs[0].reset()
        done = False
        cumu_return = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print(action.shape)
            obs, reward, terminated, truncated, info = eval_env.envs[0].step(action)
            done = terminated or truncated
            cumu_return += reward
            #print(f"Iter: {env.iter}; Reward: {env.reward:.4f} (Goal Reward: {info['goal_reward']:.4f} Guiding Reward: {info['guiding_reward']:.4f})")

        print(f"Return: {cumu_return}")

        # Save results

        frame_reached = eval_env.render("rgb_array")

        fig, axes = plt.subplots(1, 3, figsize=(30, 20))
        axes[0].set_title("Start Config", fontsize=24, fontweight="bold")
        axes[0].imshow(frame_s)
        axes[0].axis("off")
        axes[1].set_title("Target Config", fontsize=24, fontweight="bold")
        axes[1].imshow(frame_e)
        axes[1].axis("off")
        axes[2].set_title("Reached Config", fontsize=24, fontweight="bold")
        axes[2].imshow(frame_reached)
        axes[2].axis("off")
        plt.tight_layout()    
        plt.savefig(os.path.join(f"eval.png"))
        

if __name__ == "__main__":
    main()
