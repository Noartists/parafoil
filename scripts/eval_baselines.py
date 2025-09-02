import argparse
import numpy as np

from env import make_env
from baselines.controllers import PIDController, ADRCYawSpeed


def rollout(env, controller, episodes=5, max_steps=2000):
    metrics = []
    for ep in range(episodes):
        obs, _ = env.reset()
        controller.reset()
        e_p_sum = 0.0
        e_v_sum = 0.0
        e_psi_sum = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            act = controller.act(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            e_p_sum += float(np.linalg.norm(obs[0:3]))
            e_v_sum += float(np.linalg.norm(obs[3:6]))
            e_psi_sum += abs(float(obs[6]))
            steps += 1
            done = terminated or truncated
        metrics.append({
            "e_p_rmse": (e_p_sum / max(steps, 1)) ** 0.5,
            "e_v_rmse": (e_v_sum / max(steps, 1)) ** 0.5,
            "e_psi_mean": e_psi_sum / max(steps, 1),
            "steps": steps,
            "event": info.get("event", "none") if isinstance(info, dict) else "none",
        })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="pid", choices=["pid", "adrc"]) 
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_env()
    ctrl = PIDController() if args.controller == "pid" else ADRCYawSpeed()
    metrics = rollout(env, ctrl, episodes=args.episodes)
    for i, m in enumerate(metrics):
        print(f"Ep{i}: {m}")


if __name__ == "__main__":
    main()
