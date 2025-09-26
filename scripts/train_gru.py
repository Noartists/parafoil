import os
import sys
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam

# Add repo root
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env import make_env
from ppo.gru_agent import GRUActorCritic


@dataclass
class Config:
    total_timesteps: int = 500_000
    ref: str = "line"
    wind: str = "ou"
    seed: int = 0
    device: str = "cuda"
    # Environment parallelization
    n_envs: int = 4  # Number of parallel environments
    # rollout/training
    n_steps: int = 512
    epochs: int = 10
    minibatch_size: int = 128
    gamma: float = 0.995
    lam: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    n_layers: int = 1


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--ref", type=str, default="line", choices=["line","circle"])
    parser.add_argument("--wind", type=str, default="ou", choices=["none","const","ou","sin"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Override device")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN to avoid library mismatch")
    args = parser.parse_args()

    # Optional cuDNN disable (workaround for mismatched system cuDNN)
    if args.disable_cudnn:
        th.backends.cudnn.enabled = False

    # Resolve device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if th.cuda.is_available() else "cpu"

    cfg = Config(total_timesteps=args.total_timesteps, ref=args.ref, wind=args.wind, seed=args.seed, device=device, n_envs=args.n_envs)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("run", f"gru-ppo-{cfg.ref}-{cfg.wind}-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Create multiple parallel environments
    envs = [make_env(ref_type=cfg.ref, wind_mode=cfg.wind) for _ in range(cfg.n_envs)]
    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    obs_dim = obs.shape[0]
    act_dim = envs[0].action_space.shape[0]

    net = GRUActorCritic(obs_dim, act_dim, hidden_size=cfg.hidden_size, n_layers=cfg.n_layers).to(cfg.device)
    optim = Adam(net.parameters(), lr=cfg.lr)

    # rollout storage (parallel envs)
    n_steps = cfg.n_steps
    n_envs = cfg.n_envs
    obs_buf = np.zeros((n_steps + 1, n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((n_steps + 1, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

    total_steps = 0
    episode = 0
    h = net.initial_state(batch_size=n_envs).to(cfg.device)  # hidden state for all envs
    
    # Initialize observations for all environments
    for i, obs in enumerate(obs_list):
        obs_buf[0, i] = obs

    while total_steps < cfg.total_timesteps:
        # rollout
        net.eval()
        for t in range(n_steps):
            with th.no_grad():
                # Process all environments at once
                o = th.tensor(obs_buf[t], dtype=th.float32, device=cfg.device)  # (n_envs, obs_dim)
                feat = net.forward_features(o)  # (n_envs, hidden_size)
                rnn_out, h = net.forward_rnn(feat.unsqueeze(1), h)  # (n_envs, 1, hidden_size)
                rnn_last = rnn_out[:, -1, :]  # (n_envs, hidden_size)
                mu, v = net.forward_actor_critic(rnn_last)  # (n_envs, act_dim), (n_envs,)
                a, logp, u = net.sample_action(mu)  # (n_envs, act_dim), (n_envs,), (n_envs, act_dim)
            
            # Step all environments
            for i, env in enumerate(envs):
                action = a[i].cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                obs_buf[t + 1, i] = next_obs
                act_buf[t, i] = action
                logp_buf[t, i] = logp[i].item()
                rew_buf[t, i] = reward
                val_buf[t, i] = v[i].item()
                done_buf[t, i] = float(done)
                
                if done:
                    episode += 1
                    next_obs, _ = env.reset()
                    obs_buf[t + 1, i] = next_obs
                    # Reset hidden state for this environment
                    h[:, i, :] = 0.0
            
            total_steps += n_envs
        # bootstrap value for all environments
        with th.no_grad():
            o = th.tensor(obs_buf[-1], dtype=th.float32, device=cfg.device)  # (n_envs, obs_dim)
            feat = net.forward_features(o)
            rnn_out, _ = net.forward_rnn(feat.unsqueeze(1), h)
            rnn_last = rnn_out[:, -1, :]
            _, v_last = net.forward_actor_critic(rnn_last)
            val_buf[-1] = v_last.cpu().numpy()

        # compute advantage/returns for all environments
        all_advs = []
        all_rets = []
        for i in range(n_envs):
            adv, ret = compute_gae(rew_buf[:, i], val_buf[:, i], done_buf[:, i], cfg.gamma, cfg.lam)
            all_advs.append(adv)
            all_rets.append(ret)
        
        # Flatten data from all environments
        adv_flat = np.concatenate(all_advs)
        ret_flat = np.concatenate(all_rets)
        adv_t = th.tensor((adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8), dtype=th.float32, device=cfg.device)
        ret_t = th.tensor(ret_flat, dtype=th.float32, device=cfg.device)
        obs_t = th.tensor(obs_buf[:-1].reshape(-1, obs_dim), dtype=th.float32, device=cfg.device)
        act_t = th.tensor(act_buf.reshape(-1, act_dim), dtype=th.float32, device=cfg.device)
        old_logp_t = th.tensor(logp_buf.flatten(), dtype=th.float32, device=cfg.device)

        # PPO update over epochs/minibatches
        net.train()
        total_samples = n_steps * n_envs
        idxs = np.arange(total_samples)
        for epoch in range(cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, total_samples, cfg.minibatch_size):
                mb_idx = idxs[start:start + cfg.minibatch_size]
                # Build sequences for GRU: here we treat each time-step independently with single-step sequence
                o_mb = obs_t[mb_idx]
                a_mb = act_t[mb_idx]
                adv_mb = adv_t[mb_idx]
                ret_mb = ret_t[mb_idx]
                old_logp_mb = old_logp_t[mb_idx]

                feat = net.forward_features(o_mb)
                # Single-step sequence, reset hidden state per sample
                h0 = net.initial_state(batch_size=feat.shape[0]).to(cfg.device)
                rnn_out, _ = net.forward_rnn(feat.unsqueeze(1), h0)
                rnn_last = rnn_out[:, -1, :]
                mu, v = net.forward_actor_critic(rnn_last)

                logp = net.log_prob(mu, a_mb)
                ratio = th.exp(logp - old_logp_mb)
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * th.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (ret_mb - v).pow(2).mean()
                ent = th.zeros(())  # entropy term omitted for simplicity

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optim.step()

        # simple progress
        if (total_steps // (n_steps * n_envs)) % 1 == 0:
            print(f"steps={total_steps} episodes={episode} last_return={ret_flat.mean():.3f} n_envs={n_envs}")

    # save minimal artifacts
    th.save(net.state_dict(), os.path.join(run_dir, 'model_gru.pt'))
    print(f"Saved to {run_dir}")


if __name__ == '__main__':
    main()
