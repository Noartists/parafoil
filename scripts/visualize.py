import os
import numpy as np
import matplotlib.pyplot as plt


def plot_traj(run_dir: str, step_dir: str, rollout: dict):
    x = np.array(rollout["x"]) ; y = np.array(rollout["y"]) ; z = np.array(rollout["z"])
    rx = np.array(rollout["ref_x"]) ; ry = np.array(rollout["ref_y"]) ; rz = np.array(rollout["ref_z"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rx, ry, 'k--', label='ref')
    ax[0].plot(x, y, 'b-', label='traj')
    ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]'); ax[0].legend(); ax[0].set_title('XY ground track')
    t = np.array(rollout["t"]) ;
    ax[1].plot(t, rz, 'k--', label='ref z')
    ax[1].plot(t, z, 'b-', label='z')
    ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('z [m]'); ax[1].legend(); ax[1].set_title('Altitude vs time')
    os.makedirs(step_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, 'traj.png'), dpi=150)
    plt.close(fig)


def plot_errors(run_dir: str, step_dir: str, rollout: dict):
    t = np.array(rollout["t"]) ;
    epx = np.array(rollout["e_px"]) ; epy = np.array(rollout["e_py"]) ; epz = np.array(rollout["e_pz"]) ;
    epsi = np.array(rollout["e_psi"]) ;
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, epx, label='ex'); ax[0].plot(t, epy, label='ey'); ax[0].plot(t, epz, label='ez')
    ax[0].set_ylabel('pos error [m]'); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(t, epsi, label='yaw error')
    ax[1].set_ylabel('yaw err [rad]'); ax[1].set_xlabel('t [s]'); ax[1].grid(True)
    os.makedirs(step_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, 'errors.png'), dpi=150)
    plt.close(fig)


def plot_controls(run_dir: str, step_dir: str, rollout: dict):
    t = np.array(rollout["t"]) ;
    thr = np.array(rollout["thr"]) ; left = np.array(rollout["left"]) ; right = np.array(rollout["right"]) ;
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, thr) ; ax[0].set_ylabel('thrust [N]') ; ax[0].grid(True)
    ax[1].plot(t, left) ; ax[1].set_ylabel('left') ; ax[1].grid(True)
    ax[2].plot(t, right) ; ax[2].set_ylabel('right'); ax[2].set_xlabel('t [s]'); ax[2].grid(True)
    os.makedirs(step_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, 'controls.png'), dpi=150)
    plt.close(fig)

