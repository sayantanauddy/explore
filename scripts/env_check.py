import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
from gymnasium import spaces
import hydra
from omegaconf import DictConfig

from explore.env.stable_configs_env_her import StableConfigsHEREnv
from explore.env.stable_configs_env import StableConfigsEnv


def check_her_compatibility(env):
    """
    Checks key properties of an env that needs to work with SB3 + HER (HerReplayBuffer).
    """
    print("Running generic env checker...")
    print(env)
    check_env(env, warn=True, skip_render_check=True)
    print("Generic env checks passed (if no errors above).")

    # Check observation space type
    obs_space = env.observation_space
    if not isinstance(obs_space, spaces.Dict):
        print("❌ observation_space is not spaces.Dict; HER requires Dict with keys observation, achieved_goal, desired_goal.")
    else:
        keys = set(obs_space.spaces.keys())
        expected = {"observation", "achieved_goal", "desired_goal"}
        if not expected.issubset(keys):
            print(f"❌ observation_space.Dict keys are {keys}, expected at least {expected}.")
        else:
            print("✅ observation_space.Dict has required keys.")

    # Test reset output
    obs, info = env.reset(seed=0, options={})
    if not isinstance(obs, dict):
        print(f"❌ reset() returned {type(obs)}; expected dict.")
    else:
        for key in expected:
            if key not in obs:
                print(f"❌ reset() missing key '{key}'.")
            else:
                v = obs[key]
                print(f"{key}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

    if "is_success" in info:
        print("✅ info dict from reset() contains 'is_success' (optional but good).")
    else:
        print("⚠️ info dict from reset() does not contain 'is_success'. At least step() must include it eventually.")

    # Test a single step with random action
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print("After one step:")
    if not isinstance(obs2, dict):
        print(f"❌ step() returned obs of type {type(obs2)}; expected dict.")
    else:
        for key in expected:
            if key not in obs2:
                print(f"❌ step() missing key '{key}'.")
            else:
                v = obs2[key]
                print(f"{key}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

    print(f"reward type={type(reward)}, terminated={terminated}, truncated={truncated}, info keys={list(info2.keys())}")
    if "is_success" in info2:
        print("✅ info dict from step() contains 'is_success'.")
    else:
        print("⚠️ info dict from step() does not contain 'is_success'. It should for HER.")
    
    print("✔️ Compatibility check done. Interpret the above messages and fix any ❌ issues.")

# Example usage (replace MyEnv with your env class)
@hydra.main(version_base="1.3", config_path="../configs", config_name="guided_RL")
def main(cfg: DictConfig):

    def make_env():
        return StableConfigsHEREnv(cfg.env)

    env = make_vec_env(make_env, n_envs=4, env_kwargs=None)
    env0 = make_env()

    assert isinstance(
        env0, gym.Env
    ), "Your environment must inherit from the gymnasium.Env class cf. https://gymnasium.farama.org/api/env/"

    check_env(env0)


if __name__ == "__main__":
    main()
