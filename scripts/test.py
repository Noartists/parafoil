import os
import argparse
import numpy as np

from env import make_env


def run_eval(model_path: str, vecnorm_path: str, episodes: int = 3):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as e:
        print("Stable-Baselines3 missing. Install with:\n  pip install stable-baselines3[extra] gymnasium\n" 
              f"Import error: {e}")
        return

    env = DummyVecEnv([lambda: make_env()])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rew += float(reward)
            ep_len += 1
        print(f"Episode {ep}: len={ep_len}, return={ep_rew:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory containing model.zip and vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model_path = os.path.join(args.run_dir, "model.zip")
    vecnorm_path = os.path.join(args.run_dir, "vecnorm.pkl")
    run_eval(model_path, vecnorm_path, episodes=args.episodes)


if __name__ == "__main__":
    main()

