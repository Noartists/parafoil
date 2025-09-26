import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_traj(run_dir: str, step_dir: str, rollout: dict):
    x = np.array(rollout["x"]) ; y = np.array(rollout["y"]) ; z = np.array(rollout["z"])
    rx = np.array(rollout["ref_x"]) ; ry = np.array(rollout["ref_y"]) ; rz = np.array(rollout["ref_z"])
    
    # Create figure with 3 subplots: 2D XY, altitude vs time, and 3D trajectory
    fig = plt.figure(figsize=(15, 5))
    
    # 2D XY plot
    ax1 = fig.add_subplot(131)
    ax1.plot(rx, ry, 'k--', label='ref', linewidth=2)
    ax1.plot(x, y, 'b-', label='traj', linewidth=1.5)
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.legend(); ax1.set_title('XY ground track')
    ax1.grid(True, alpha=0.3)
    
    # Altitude vs time
    t = np.array(rollout["t"])
    ax2 = fig.add_subplot(132)
    ax2.plot(t, rz, 'k--', label='ref z', linewidth=2)
    ax2.plot(t, z, 'b-', label='z', linewidth=1.5)
    ax2.set_xlabel('t [s]'); ax2.set_ylabel('z [m]'); ax2.legend(); ax2.set_title('Altitude vs time')
    ax2.grid(True, alpha=0.3)
    
    # 3D trajectory plot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(rx, ry, rz, 'k--', label='ref', linewidth=2)
    ax3.plot(x, y, z, 'b-', label='traj', linewidth=1.5)
    ax3.scatter(x[0], y[0], z[0], color='green', s=50, label='start')
    ax3.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='end')
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.legend(); ax3.set_title('3D trajectory')

    # Set equal aspect ratio for better visualization
    # Get the ranges for each axis
    x_range = np.ptp(np.concatenate([x, rx]))  # peak-to-peak (max - min)
    y_range = np.ptp(np.concatenate([y, ry]))
    z_range = np.ptp(np.concatenate([z, rz]))

    # Use the maximum range to set all axes to same scale
    max_range = max(x_range, y_range, z_range)

    # Get the center points
    x_center = (np.max(np.concatenate([x, rx])) + np.min(np.concatenate([x, rx]))) / 2
    y_center = (np.max(np.concatenate([y, ry])) + np.min(np.concatenate([y, ry]))) / 2
    z_center = (np.max(np.concatenate([z, rz])) + np.min(np.concatenate([z, rz]))) / 2

    # Set the limits to create equal aspect ratio
    ax3.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax3.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax3.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
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

