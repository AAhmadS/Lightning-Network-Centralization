import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomTransformer(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, embed_dim: int = 128,
                  nhead: int = 4, num_layers: int = 3):
        super().__init__(observation_space, features_dim)
        # We assume [batch_size, sequence_size, num_node_features]
        # print("OBS:", observation_space.shape)
        num_node_features = observation_space.shape[1]
        sequence_size = observation_space.shape[0]

        # Embedding layer for node features
        self.embedding = nn.Linear(num_node_features, embed_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # The output dimension after the transformer and flattening
        n_flatten = sequence_size * embed_dim
        # n_flatten = embed_dim

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size, sequence_size, num_node_features = observations.shape
        embedded_obs = self.embedding(observations.view(batch_size, sequence_size, num_node_features))
        transformed_obs = self.transformer(embedded_obs)
        flattened_obs = transformed_obs.view(batch_size, -1)
        return self.linear(flattened_obs)

