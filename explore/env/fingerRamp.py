import os
import h5py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from gymnasium_robotics.core import GoalEnv
from pprint import pprint

from explore.env.mujoco_sim import MjSim
from explore.utils.utils import randint_excluding

class FingerRampHerEnv(GoalEnv, gym.Env):  # GoalEnv needed for HER

    def __init__(self, cfg: DictConfig):
        
        super().__init__()

        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        self.mujoco_xml = cfg.sim.mujoco_xml

        self.stepsize = cfg.env.stepsize
        self.max_steps_default = cfg.env.max_steps 
        
        self.actions_noise_sigma = cfg.env.actions_noise_sigma
        self.use_vel = cfg.env.use_vel
        
        self.use_vision = cfg.env.use_vision
        assert (self.use_vision and not self.use_vel) or (not self.use_vision and self.use_vel)

        # Setup sim
        self.start_config_idx = cfg.env.start_config_idx
        self.end_config_idx = cfg.env.target_config_idx
        self.goal_conditioning = cfg.env.goal_conditioning

        if not self.goal_conditioning and cfg.env.target_config_idx == -1:
            raise Exception("Setting many goals but no goal conditioning!")
        if self.goal_conditioning and cfg.env.target_config_idx != -1:
            raise Exception("Using goal conditioning but only a single goal!")
        
        self.stable_configs = h5py.File(cfg.env.stable_configs_path, 'r')
        self.config_count = self.stable_configs["qpos"].shape[0]
        
        self.verbose = cfg.verbose


        # TODO: What is the purpose of this
        self.q_mask = np.array(cfg.q_mask)

        self.time_scaling = cfg.env.time_scaling
            
        # Set up sim
        self.sim = MjSim(
            self.mujoco_xml, 
            self.tau_sim,
            interpolate=self.interpolate_actions, 
            joints_are_same_as_ctrl=self.joints_are_same_as_ctrl
            )
        self.sim.setupRenderer(160, 160, camera=cfg.sim.camera)
        
        # Define observation space
        state = self.sim.getState()
        state_n = state[1].shape[0]  # qpos
        self.state_dim = state_n
        if self.use_vel:
            state_n += state[2].shape[0]  # qvel
        if self.goal_conditioning:
            state_n += state[1].shape[0]
        
        self.ctrl_dim = self.sim.data.ctrl.shape[0]

        if self.use_vision: # will throw error
            raise NotImplementedError("Vision is not supported atm")
        else:
            # HER needs dict observations
            self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=-np.inf, high=np.inf, shape=(state_n,), dtype=np.float32),
                achieved_goal=spaces.Box(low=-np.inf, high=np.inf, shape=(state[1].shape[0],), dtype=np.float32),
                desired_goal=spaces.Box(low=-np.inf, high=np.inf, shape=(state[1].shape[0],), dtype=np.float32),
            )
        )

        # TODO: Better to have actions in [-1,1]
        # Defines action space
        if self.stepsize != -1:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0].astype(np.float32)
            max_ctrl = ctrl_ranges[:, 1].astype(np.float32)
        
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(self.ctrl_dim,), dtype=np.float32)
        
        
    def getState(self) -> np.ndarray:
        
        self.sim_state = self.sim.getState()
        time_, qpos, qvel, ctrl = self.sim_state
        
        if self.use_vision:

            # HER with images is tricky to implement
            # HerReplayBuffer expects vector goals, not images (we still need numeric goals)
            # Latent features can possibly be used: https://proceedings.neurips.cc/paper/2018/file/7ec69dd44416c46745f6edd947b470cd-Paper.pdf
            raise NotImplementedError("Vision is not supported atm")
                        
        else:
            self.state = qpos
            if self.use_vel:
                self.state = np.concatenate((self.state, qvel))
            
            if self.goal_conditioning:
                self.state = np.concatenate((self.state, self.target_state[:self.state_dim]))

        self.last_time = time_
        self.last_ctrl = ctrl
        return self.state.astype(np.float32)

    def reset(self, *, seed: int=None, options: dict={}):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.eval_view = "" if not "eval_view" in options else options["eval_view"]

        # Choose start and end configurations
        s_cfg_idx = self.start_config_idx if self.start_config_idx != -1 else np.random.randint(0, self.config_count)
        e_cfg_idx = self.end_config_idx if self.end_config_idx != -1 else randint_excluding(0, self.config_count, s_cfg_idx)
            
        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx}
        
        self.target_state = self.stable_configs["qpos"][e_cfg_idx]
        
        # Reset simulation state
        self.sim.pushConfig(
            self.stable_configs["qpos"][s_cfg_idx],
            self.stable_configs["ctrl"][s_cfg_idx]
        )
        
        self.iter = 0
        
        self.max_steps = self.max_steps_default
        
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        obs_dict = {
            "observation": self.getState(), # observation shape is (29,0)
            "achieved_goal": self.sim_state[1].copy(),
            "desired_goal": self.target_state.copy(),
        }

        return obs_dict, info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if self.actions_noise_sigma != -1:
            action += np.random.randn(action.shape[-1]) * self.actions_noise_sigma
        
        if self.stepsize != -1:
            action += self.last_ctrl
        
        frames = self.sim.step(self.tau_action, action, view=self.eval_view)

        self.getState()
        self.iter += 1

        ### Reward Computation ###
        self.eval_state = self.sim_state[1]
        
        desired_goal = self.target_state.copy()
        achieved_goal = self.eval_state.copy()

        info = {
            "frames": frames,
        }

        # Needed by HER
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        info["goal_reward"] = reward

        # Needs to change
        truncated = self.iter >= self.max_steps
        terminated = truncated

        obs_dict = {
            "observation": self.getState().astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

        return obs_dict, reward, terminated, truncated, info
        
    def compute_reward(    
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict
        ) -> float | np.ndarray:

        nenv = achieved_goal.shape[0]

        # Vectorized code
        e_goal = achieved_goal*self.q_mask - desired_goal*self.q_mask  # shape: (n_env, ndim) or (ndim,)
        goal_reward = -np.sum(e_goal**2, axis=-1) # Can handle vectorized and single env
        
        # Clamping reward
        goal_reward = np.maximum(goal_reward, -5.0)

        return goal_reward

    def render(self, mode: str="", config_idx: int=-1) -> np.ndarray:
        # Separate scene renderer from model vision

        if config_idx != -1:
            current_state = self.sim.getState()
            self.sim.setState(*self.trees[config_idx][0]["state"])

        if mode:
            img = self.sim.renderImg(mode)
        else:
            img = self.sim.renderImg()

        if config_idx != -1:
            self.sim.setState(*current_state)

        return img
