import os
import json
import argparse
from datetime import datetime

from env import make_env
from scripts.callbacks import CustomEvalCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--ref", type=str, default="line", choices=["line", "circle"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Runtime dirs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("run", f"ppo-{args.ref}-{ts}")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(run_dir, exist_ok=True)

    # Lazy import SB3
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    except Exception as e:
        print("Stable-Baselines3 missing. Install with:\n  pip install stable-baselines3[extra] gymnasium\n" 
              f"Import error: {e}")
        return

    # Env factory
    def make_one(rank: int):
        def _thunk():
            env = make_env(ref_type=args.ref)
            env = Monitor(env)
            return env
        return _thunk

    n_envs = 1
    try:
        import os as _os
        n_envs = max(1, _os.cpu_count() // 2)
    except Exception:
        pass

    vec_env = SubprocVecEnv([make_one(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, clip_obs=10.0, clip_reward=10.0)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
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

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "n_envs": n_envs}, f, indent=2)

    # Evaluation callback: save plots/metrics into run_dir
    eval_cb = CustomEvalCallback(
        make_env_fn=lambda: make_env(ref_type=args.ref),
        run_dir=run_dir,
        eval_freq=10000,
        n_eval_episodes=2,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)

    # Save artifacts into run_dir
    model.save(os.path.join(run_dir, "model.zip"))
    vec_env.save(os.path.join(run_dir, "vecnorm.pkl"))
    print(f"Saved artifacts to {run_dir}")


if __name__ == "__main__":
    main()
