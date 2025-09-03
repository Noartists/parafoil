import os
import time
import argparse

from env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between steps for slow play")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as e:
        print("Stable-Baselines3 missing. Install with:\n  pip install stable-baselines3[extra] gymnasium\n" 
              f"Import error: {e}")
        return

    model_path = os.path.join(args.run_dir, "model.zip")
    vecnorm_path = os.path.join(args.run_dir, "vecnorm.pkl")

    env = DummyVecEnv([lambda: make_env()])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    done_flag = False
    step = 0
    while step < args.max_steps and not done_flag:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        if args.sleep > 0:
            time.sleep(args.sleep)
        try:
            done_flag = bool(dones[0])
        except Exception:
            done_flag = bool(dones)
        step += 1
    if done_flag:
        print("Episode finished.")


if __name__ == "__main__":
    main()
