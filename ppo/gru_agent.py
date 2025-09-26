import math
from typing import Tuple

import torch as th
from torch import nn


def atanh(x: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (th.log1p(x) - th.log1p(-x))


class GRUActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128, n_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, hidden_size), nn.Tanh(),
        )
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.actor_mu = nn.Linear(hidden_size, act_dim)
        self.actor_logstd = nn.Parameter(th.zeros(act_dim))
        self.critic = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(m.bias)

    def initial_state(self, batch_size: int = 1) -> th.Tensor:
        return th.zeros(self.n_layers, batch_size, self.hidden_size)

    def forward_features(self, obs: th.Tensor) -> th.Tensor:
        return self.encoder(obs)

    def forward_rnn(self, feat_seq: th.Tensor, h: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # feat_seq: (B, T, F), h: (L, B, H)
        out, h_new = self.gru(feat_seq, h)
        return out, h_new

    def forward_actor_critic(self, rnn_out: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mu = self.actor_mu(rnn_out)
        v = self.critic(rnn_out).squeeze(-1)
        return mu, v

    def dist(self, mu: th.Tensor) -> th.distributions.Normal:
        std = th.exp(self.actor_logstd)
        return th.distributions.Normal(mu, std)

    def sample_action(self, mu: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Reparameterized sample: pre-tanh u ~ N(mu, std), a = tanh(u)
        dist = self.dist(mu)
        u = mu + th.exp(self.actor_logstd) * th.randn_like(mu)
        a = th.tanh(u)
        # log prob with tanh correction
        log_prob = dist.log_prob(u) - th.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return a, log_prob, u

    def log_prob(self, mu: th.Tensor, a: th.Tensor) -> th.Tensor:
        # Given action a in [-1,1], compute log prob under current policy
        u = atanh(a)
        dist = self.dist(mu)
        log_prob = dist.log_prob(u) - th.log(1 - a.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

