import os
import json
import argparse
from datetime import datetime

from env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--ref", type=str, default="line", choices=["line", "circle"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wind", type=str, default="ou", choices=["none","const","ou","sin"])
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("run", f"rppo-{args.ref}-{args.wind}-{ts}")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(run_dir, exist_ok=True)

    try:
        from sb3_contrib import RecurrentPPO
        from ppo.gru_policy import MlpGruPolicy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    except Exception as e:
        print("Install sb3-contrib for RecurrentPPO:\n  pip install sb3-contrib stable-baselines3[extra] gymnasium\n"
              f"Import error: {e}")
        return

    def make_one(rank: int):
        def _thunk():
            env = make_env(ref_type=args.ref, wind_mode=args.wind)
            env = Monitor(env)
            return env
        return _thunk

    n_envs = 4
    vec_env = SubprocVecEnv([make_one(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, clip_obs=10.0, clip_reward=10.0)

    policy_kwargs = dict(
        gru_hidden_size=128,
        n_gru_layers=1,
        shared_lstm=True,
    )
    model = RecurrentPPO(
        MlpGruPolicy,
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_dir,
    )

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "algo": "RecurrentPPO"}, f, indent=2)

    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(run_dir, "model.zip"))
    vec_env.save(os.path.join(run_dir, "vecnorm.pkl"))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
