import os
import csv
from typing import Callable, List

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import make_env
from scripts.visualize import plot_traj, plot_errors, plot_controls


class CustomEvalCallback(BaseCallback):
    """Evaluate policy periodically, save metrics/rollouts/plots under run_dir.

    Copies VecNormalize stats from training env for fair evaluation.
    """

    def __init__(
        self,
        make_env_fn: Callable[[], object],
        run_dir: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True

        # Build eval env and copy VecNormalize stats
        train_env = self.model.get_env()
        eval_env = DummyVecEnv([self.make_env_fn])
        if isinstance(train_env, VecNormalize):
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            eval_env.clip_obs = train_env.clip_obs
            eval_env.clip_reward = train_env.clip_reward

        step_dir = os.path.join(self.run_dir, 'eval', f'step_{self.num_timesteps}')
        os.makedirs(step_dir, exist_ok=True)

        metrics = []
        for ep in range(self.n_eval_episodes):
            rollout = self._rollout_one(eval_env)
            # Save CSV
            csv_path = os.path.join(step_dir, f'rollout_ep{ep}.csv')
            self._save_rollout_csv(csv_path, rollout)
            # Plots
            plot_traj(self.run_dir, step_dir, rollout)
            plot_errors(self.run_dir, step_dir, rollout)
            plot_controls(self.run_dir, step_dir, rollout)
            # Metrics
            ep_metrics = self._compute_metrics(rollout)
            metrics.append(ep_metrics)

        # Aggregate and write metrics.csv (append)
        agg = {
            'timesteps': self.num_timesteps,
            'e_p_rmse_mean': float(np.mean([m['e_p_rmse'] for m in metrics])),
            'e_psi_mean': float(np.mean([m['e_psi_mean'] for m in metrics])),
            'len_mean': float(np.mean([m['len'] for m in metrics])),
        }
        self._append_metrics(os.path.join(self.run_dir, 'eval', 'metrics.csv'), agg)

        # Also log to tensorboard if available
        try:
            self.logger.record('eval/e_p_rmse_mean', agg['e_p_rmse_mean'])
            self.logger.record('eval/e_psi_mean', agg['e_psi_mean'])
            self.logger.record('eval/len_mean', agg['len_mean'])
        except Exception:
            pass

        return True

    def _rollout_one(self, vec_env) -> dict:
        obs = vec_env.reset()
        done = False
        data = {k: [] for k in ['t','x','y','z','ref_x','ref_y','ref_z','phi','theta','psi','vx','vy','vz','thr','left','right','e_px','e_py','e_pz','e_psi']}
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)
            # Access underlying raw env
            raw = vec_env.venv.envs[0]
            try:
                # For VecNormalize, raw is the underlying DummyVecEnv; take its envs[0]
                if isinstance(raw, VecNormalize):
                    raw = raw.venv.envs[0]
            except Exception:
                pass
            e = raw
            y = e.state
            t = e.t
            px, py, pz = y[0:3]
            phi, theta, psi = y[3:6]
            vx, vy, vz = y[8:11]
            ref_p, ref_v, ref_yaw = e._ref(t)
            epx, epy, epz = (px - ref_p[0]), (py - ref_p[1]), (pz - ref_p[2])
            epsi = ((psi - ref_yaw + np.pi) % (2*np.pi)) - np.pi
            thr, left, right = float(e._last_cmd[0]), float(e._last_cmd[1]), float(e._last_cmd[2])
            data['t'].append(float(t))
            data['x'].append(float(px)); data['y'].append(float(py)); data['z'].append(float(pz))
            data['ref_x'].append(float(ref_p[0])); data['ref_y'].append(float(ref_p[1])); data['ref_z'].append(float(ref_p[2]))
            data['phi'].append(float(phi)); data['theta'].append(float(theta)); data['psi'].append(float(psi))
            data['vx'].append(float(vx)); data['vy'].append(float(vy)); data['vz'].append(float(vz))
            data['thr'].append(thr); data['left'].append(left); data['right'].append(right)
            data['e_px'].append(float(epx)); data['e_py'].append(float(epy)); data['e_pz'].append(float(epz)); data['e_psi'].append(float(epsi))
        return data

    @staticmethod
    def _compute_metrics(rollout: dict) -> dict:
        e_p = np.sqrt(np.array(rollout['e_px'])**2 + np.array(rollout['e_py'])**2 + np.array(rollout['e_pz'])**2)
        e_psi = np.abs(np.array(rollout['e_psi']))
        return {
            'e_p_rmse': float(np.sqrt(np.mean(e_p**2))),
            'e_psi_mean': float(np.mean(e_psi)),
            'len': int(len(rollout['t'])),
        }

    @staticmethod
    def _save_rollout_csv(path: str, data: dict):
        keys = list(data.keys())
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            rows = zip(*[data[k] for k in keys])
            for r in rows:
                writer.writerow(r)

    @staticmethod
    def _append_metrics(path: str, metrics: dict):
        exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(metrics)

