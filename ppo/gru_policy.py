from typing import Optional, Tuple

import torch as th
from torch import nn

try:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except Exception as e:  # pragma: no cover
    RecurrentActorCriticPolicy = object  # to avoid linter issues if not installed


class MlpGruPolicy(RecurrentActorCriticPolicy):
    """Recurrent Actor-Critic with GRU backbone (batch_first).

    This mirrors the structure of MlpLstmPolicy but replaces LSTM with GRU.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        gru_hidden_size: int = 128,
        n_gru_layers: int = 1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.gru_hidden_size = gru_hidden_size
        self.n_gru_layers = n_gru_layers

        # Replace default lstm with GRU
        features_dim = self.features_extractor.features_dim
        self.rnn = nn.GRU(
            input_size=features_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.n_gru_layers,
            batch_first=True,
        )

        # Update mlp_extractor input dim: use GRU hidden
        # self.mlp_extractor created in parent, but depends on features_dim.
        # Recreate with correct input dim.
        self._build_mlp_extractor()

        # Action and value nets already defined in parent (depend on mlp_extractor output)

        # Initialize parameters
        self._initialize_weights()

    def _build_mlp_extractor(self) -> None:
        # Rebuild MLP extractor with GRU hidden size as input
        # Using same net_arch as parent kwargs
        self.mlp_extractor = self.mlp_extractor_class(
            self.gru_hidden_size, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

    def _initialize_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward for one step (used during sampling).

        obs: (batch, obs_dim)
        lstm_states: (n_layers, batch, hidden)
        episode_starts: (batch,)
        Returns: actions, values, new_states
        """
        features = self.extract_features(obs)
        # Reset states where episode starts
        if lstm_states is None:
            lstm_states = th.zeros(self.n_gru_layers, obs.shape[0], self.gru_hidden_size, device=obs.device)
        mask = (1.0 - episode_starts.float()).view(1, -1, 1)
        lstm_states = lstm_states * mask

        rnn_output, new_states = self.rnn(features.unsqueeze(1), lstm_states)
        rnn_output = rnn_output.squeeze(1)

        latent_pi, latent_vf = self.mlp_extractor(rnn_output)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, new_states, log_prob

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: th.Tensor,
        episode_starts: th.Tensor,
    ):
        features = self.extract_features(obs)
        if lstm_states is None:
            lstm_states = th.zeros(self.n_gru_layers, obs.shape[0], self.gru_hidden_size, device=obs.device)
        mask = (1.0 - episode_starts.float()).view(1, -1, 1)
        lstm_states = lstm_states * mask
        rnn_output, new_states = self.rnn(features.unsqueeze(1), lstm_states)
        rnn_output = rnn_output.squeeze(1)
        latent_pi, _ = self.mlp_extractor(rnn_output)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution, new_states

    def forward_actor(self, obs: th.Tensor, lstm_states: th.Tensor, episode_starts: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if lstm_states is None:
            lstm_states = th.zeros(self.n_gru_layers, obs.shape[0], self.gru_hidden_size, device=obs.device)
        mask = (1.0 - episode_starts.float()).view(1, -1, 1)
        lstm_states = lstm_states * mask
        rnn_output, _ = self.rnn(features.unsqueeze(1), lstm_states)
        rnn_output = rnn_output.squeeze(1)
        latent_pi, _ = self.mlp_extractor(rnn_output)
        return latent_pi

    def forward_critic(self, obs: th.Tensor, lstm_states: th.Tensor, episode_starts: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if lstm_states is None:
            lstm_states = th.zeros(self.n_gru_layers, obs.shape[0], self.gru_hidden_size, device=obs.device)
        mask = (1.0 - episode_starts.float()).view(1, -1, 1)
        lstm_states = lstm_states * mask
        rnn_output, _ = self.rnn(features.unsqueeze(1), lstm_states)
        rnn_output = rnn_output.squeeze(1)
        _, latent_vf = self.mlp_extractor(rnn_output)
        values = self.value_net(latent_vf)
        return values

