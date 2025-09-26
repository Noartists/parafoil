import argparse
import os
import sys
from datetime import datetime
import numpy as np

# Add repo root to sys.path for imports when running as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import make_env
from typing import Optional, Dict
from baselines.controllers import PIDController, ADRCYawSpeed
from scripts.visualize import plot_traj, plot_errors, plot_controls


def rollout(env, controller, max_steps=1000, init: Optional[Dict] = None, debug=False):
    if init is not None:
        obs, _ = env.reset(options={"init": init})
    else:
        obs, _ = env.reset()
    controller.reset()
    done = False
    data = {k: [] for k in ['t','x','y','z','ref_x','ref_y','ref_z','phi','theta','psi','vx','vy','vz','thr','left','right','e_px','e_py','e_pz','e_psi']}
    steps = 0
    while not done and steps < max_steps:
        act = controller.act(obs)
        if debug and steps < 5:
            print(f"Step {steps}: act={act}")
        obs, rew, terminated, truncated, info = env.step(act)
        y = env.state
        if debug and steps % 100 == 0:
            print(f"Step {steps}, pos: [{y[0]:.1f}, {y[1]:.1f}, {y[2]:.1f}], speed: {np.linalg.norm(y[8:11]):.1f}")
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
    # Initial condition options (all optional)
    parser.add_argument("--init-pos", type=str, default=None, help="x,y,z")
    parser.add_argument("--init-att", type=str, default=None, help="phi,theta,psi [rad]")
    parser.add_argument("--init-v", type=str, default=None, help="vx,vy,vz [m/s]")
    parser.add_argument("--init-w", type=str, default=None, help="wx,wy,wz [rad/s]")
    parser.add_argument("--init-vp", type=str, default=None, help="vpx,vpy,vpz [m/s]")
    parser.add_argument("--init-ws", type=str, default=None, help="wsx,wsy,wsz [rad/s]")
    parser.add_argument("--init-thetar", type=float, default=None)
    parser.add_argument("--init-psir", type=float, default=None)
    parser.add_argument("--init-throttle", type=float, default=None, help="[N]")
    parser.add_argument("--init-left", type=float, default=None, help="deflection")
    parser.add_argument("--init-right", type=float, default=None, help="deflection")
    args = parser.parse_args()

    # Small scale parafoil actuator limits (from MATLAB model)
    action_limits = {
        "thrust_max": 30.0,        # N (small scale thrust limit)
        "deflection_max": 0.3,     # rad (small scale brake deflection)  
        "deflection_rate_max": 3.0, # rad/s (brake actuation speed)
        "thrust_rate_max": 100.0   # N/s (engine response)
    }
    env = make_env(ref_type=args.ref, wind_mode=args.wind, action_limits=action_limits, actuator_tau=0.05)
    # Adjust initial position to match reference trajectory start
    init_config = INIT_DEFAULT.copy()
    if args.ref == "circle":
        # Circle starts at (R, 0, alt) where R=120m 
        init_config['pos'] = [120.0, 0.0, 230.0]
        # Initial velocity should match circle tangent direction (0, +speed, vz)
        init_config['v'] = [0.0, 8.5, 1.2]  # tangent to circle at t=0
        # Initial yaw should be π/2 (90 degrees, pointing north)
        init_config['att'] = [0.0, 0.0, 1.5708]  # π/2 radians
    else:
        # Line starts at (0, 0, alt)  
        init_config['pos'] = [0.0, 0.0, 230.0]
    env.set_default_init(init_config)
    if args.controller == "pid":
        ctrl = PIDController()
    else:
        ctrl = ADRCYawSpeed()

    run_dir = None
    if args.save:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join("run", f"baseline-{args.controller}-{args.ref}-{args.wind}-{ts}")
        os.makedirs(run_dir, exist_ok=True)

    # Build init dict if any args provided
    def parse_vec(s):
        if s is None:
            return None
        vals = [float(x) for x in s.split(',')]
        if len(vals) != 3:
            raise ValueError("Vector args must have 3 comma-separated values")
        return vals

    init = {}
    any_init = False
    for key, val in (
        ("pos", parse_vec(args.init_pos)),
        ("att", parse_vec(args.init_att)),
        ("v", parse_vec(args.init_v)),
        ("w", parse_vec(args.init_w)),
        ("vp", parse_vec(args.init_vp)),
        ("ws", parse_vec(args.init_ws)),
    ):
        if val is not None:
            init[key] = val
            any_init = True
    if args.init_thetar is not None:
        init["thetar"] = args.init_thetar; any_init = True
    if args.init_psir is not None:
        init["psir"] = args.init_psir; any_init = True
    act = {}
    if args.init_throttle is not None:
        act["throttle"] = args.init_throttle
    if args.init_left is not None:
        act["left"] = args.init_left
    if args.init_right is not None:
        act["right"] = args.init_right
    if len(act) > 0:
        init["act"] = act; any_init = True
    init_dict = init if any_init else None

    for ep in range(args.episodes):
        print(f"Starting episode {ep}...")
        ro = rollout(env, ctrl, init=init_dict, debug=True)
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


"""
Default initial conditions for eval runs.
Edit this dict to change your default startup state.
These values are used unless overridden by --init-* CLI flags.
"""
INIT_DEFAULT: Dict = {
    # Position [m] - matches circle ref start (R,0,alt) or line ref start (0,0,alt)
    'pos': [0.0, 0.0, 230.0],
    # Attitude [rad] (phi, theta, psi)
    'att': [0.0, 0.0, 0.0],
    # Canopy velocity [m/s] - real scale parafoil
    'v': [8.5, 0.0, 1.2],
    # Angular rates [rad/s]
    'w': [0.0, 0.0, 0.0],
    # Relative angles
    'thetar': 0.0,
    'psir': 0.0,
    # Actuator initial physical commands (small scale, cruise mode)
    'act': {'throttle': 15.0, 'left': 0.0, 'right': 0.0},  # 50% throttle for cruise
}

if __name__ == "__main__":
    main()
