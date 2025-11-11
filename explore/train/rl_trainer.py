import os
import logging
from pathlib import Path

from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

from explore.env.fingerRamp import FingerRampHerEnv


class RL_Trainer:

    def __init__(self, cfg: DictConfig, logger: logging.Logger):

        self.cfg = cfg
        self.logger = logger
        
        self.total_timesteps = cfg.total_timesteps
        self.save_as = Path(cfg.output_dir) / Path("final_rl_policy")
        self.save_freq = cfg.save_freq

        # Logginf directories
        self.base_dir = Path(cfg.output_dir) 
        self.model_dir = self.base_dir / "models"
        self.video_dir = self.base_dir / "videos"
        self.tb_dir = self.base_dir / "tensorboard"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        run = wandb.init(
            project=cfg.wandb.project,
            #name=cfg.wandb.name,
            dir=str(self.base_dir),
            sync_tensorboard=True,
            save_code=True,
            monitor_gym=True,
        )

        # Environments
        self.train_env = make_vec_env(self.make_env, n_envs=cfg.env.num_envs)
        self.eval_env = make_vec_env(self.make_env, n_envs=1)
        self.eval_env.render_mode = "rgb_array"

        # Model logger
        model_logger = configure(str(self.base_dir), ["stdout", "csv", "tensorboard"])

        # Device
        self.device = cfg.device
        self.logger.info(f"Using device: {self.device}")

        # Define model
        self.rl_method = cfg.her.rl_method

        # HER needs Dict observations
        policy = "MultiInputPolicy"

        if self.rl_method == "SAC":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=cfg.her.net_arch,
                    qf=cfg.her.net_arch
                )
            )
            self.model = SAC(
                policy,
                self.train_env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=cfg.her.n_sampled_goal,
                    goal_selection_strategy=cfg.her.strategy,
                ),
                policy_kwargs=policy_kwargs, 
                verbose=cfg.verbose, 
                device=self.device,
                batch_size=cfg.her.batch_size,
                learning_starts=cfg.her.learning_starts,  # wait until <learning_starts> transitions are collected
            )
        else:
            raise Exception(f"RL method '{self.rl_method}' not available with HER.")
        
        self.model.set_logger(model_logger)

        # Eval should have videos
        self.eval_env = VecVideoRecorder(
            self.eval_env,
            str(self.video_dir),
            record_video_trigger=lambda step: step % cfg.eval.video_interval == 0,
            video_length=cfg.eval.video_length,
            name_prefix="vid",
        )

        # --- Callbacks ---
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir),
            log_path=str(self.base_dir),
            eval_freq=cfg.eval.eval_freq,
            deterministic=True,
        )

        self.checkpoint_callback = CheckpointCallback(
            save_freq=cfg.checkpoint.save_freq,
            save_path=str(self.model_dir),
            name_prefix="checkpoint",
        )

        self.wandb_callback = WandbCallback(
            model_save_path=f"{self.model_dir}/{run.id}",
            model_save_freq=cfg.checkpoint.save_freq,
            log="all",
        )

    def make_env(self):
        return FingerRampHerEnv(self.cfg)

    def train(self):
        self.logger.info("Starting training...")

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.checkpoint_callback, self.eval_callback, self.wandb_callback]
        )

        self.model.save(self.save_as)
        self.logger.info(f"Model saved as {self.save_as}")
