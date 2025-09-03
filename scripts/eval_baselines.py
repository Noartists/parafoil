import argparse
import os
from datetime import datetime
import numpy as np

from env import make_env
from baselines.controllers import PIDController, ADRCYawSpeed
from scripts.visualize import plot_traj, plot_errors, plot_controls


def rollout(env, controller, max_steps=2000):
    obs, _ = env.reset()
    controller.reset()
    done = False
    data = {k: [] for k in ['t','x','y','z','ref_x','ref_y','ref_z','phi','theta','psi','vx','vy','vz','thr','left','right','e_px','e_py','e_pz','e_psi']}
    steps = 0
    while not done and steps < max_steps:
        act = controller.act(obs)
        obs, rew, terminated, truncated, info = env.step(act)
        y = env.state
        t = env.t
        px, py, pz = y[0:3]
        phi, theta, psi = y[3:6]
        vx, vy, vz = y[8:11]
        ref_p, ref_v, ref_yaw = env._ref(t)
        epx, epy, epz = (px - ref_p[0]), (py - ref_p[1]), (pz - ref_p[2])
        epsi = ((psi - ref_yaw + np.pi) % (2*np.pi)) - np.pi
        thr, left, right = float(env._last_cmd[0]), float(env._last_cmd[1]), float(env._last_cmd[2])
        data['t'].append(float(t))
        data['x'].append(float(px)); data['y'].append(float(py)); data['z'].append(float(pz))
        data['ref_x'].append(float(ref_p[0])); data['ref_y'].append(float(ref_p[1])); data['ref_z'].append(float(ref_p[2]))
        data['phi'].append(float(phi)); data['theta'].append(float(theta)); data['psi'].append(float(psi))
        data['vx'].append(float(vx)); data['vy'].append(float(vy)); data['vz'].append(float(vz))
        data['thr'].append(thr); data['left'].append(left); data['right'].append(right)
        data['e_px'].append(float(epx)); data['e_py'].append(float(epy)); data['e_pz'].append(float(epz)); data['e_psi'].append(float(epsi))
        steps += 1
        done = terminated or truncated
    return data


def metrics_from_rollout(rollout: dict):
    e_p = np.sqrt(np.array(rollout['e_px'])**2 + np.array(rollout['e_py'])**2 + np.array(rollout['e_pz'])**2)
    e_psi = np.abs(np.array(rollout['e_psi']))
    return {
        'e_p_rmse': float(np.sqrt(np.mean(e_p**2))),
        'e_psi_mean': float(np.mean(e_psi)),
        'len': int(len(rollout['t'])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="pid", choices=["pid", "adrc"]) 
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--ref", type=str, default="line", choices=["line", "circle"]) 
    parser.add_argument("--wind", type=str, default="none", choices=["none","const","ou","sin"]) 
    parser.add_argument("--save", action="store_true", help="Save plots and csv to run/ directory")
    args = parser.parse_args()

    env = make_env(ref_type=args.ref, wind_mode=args.wind)
    ctrl = PIDController() if args.controller == "pid" else ADRCYawSpeed()

    run_dir = None
    if args.save:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("run", f"baseline-{args.controller}-{args.ref}-{args.wind}-{ts}")
        os.makedirs(run_dir, exist_ok=True)

    for ep in range(args.episodes):
        ro = rollout(env, ctrl)
        m = metrics_from_rollout(ro)
        print(f"Ep{ep}: {m}")
        if args.save:
            step_dir = os.path.join(run_dir, f"ep_{ep}")
            os.makedirs(step_dir, exist_ok=True)
            # Save plots
            plot_traj(run_dir, step_dir, ro)
            plot_errors(run_dir, step_dir, ro)
            plot_controls(run_dir, step_dir, ro)
            # Save csv
            import csv
            keys = list(ro.keys())
            with open(os.path.join(step_dir, 'rollout.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for row in zip(*[ro[k] for k in keys]):
                    writer.writerow(row)


if __name__ == "__main__":
    main()
