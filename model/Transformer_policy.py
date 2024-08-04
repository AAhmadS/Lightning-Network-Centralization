from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import gymnasium as gym



import torch as th
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Union, List, Dict, Type, Tuple
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)
         
class NullFeatureExtractor(BaseFeaturesExtractor):
    """
    acts nothing, gives the tensor of the exact shape to the transformer.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations
    
    
class CustomTransformerExtractor(nn.Module):
    """
    Constructs a transformer-based model that receives the output from a previous features extractor
    (i.e., a CNN) or directly the observations (if no features extractor is applied) as input and outputs
    a latent representation for the policy and a value network.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    :param hidden_dim: The hidden dimension size of the transformer.
    :param num_layers: Number of transformer layers.
    :param num_heads: Number of attention heads in the transformer.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        hidden_dim = 32,
        num_heads = 4,
        num_layers = 4,
        max_position_embedding = 50,
        input_dim = 4
    ) -> None:
        super().__init__()
        
        device = get_device(device)
        
        self.embedding = nn.Linear(input_dim, hidden_dim).to(device)
        
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,  # Common practice for transformer models
            max_position_embeddings=max_position_embedding,  # Default value, adjust if needed
            position_embedding_type="none"  # Disable positional encoding
        )
        
        self.transformer = BertModel(config).to(device)
        
        self.flatten = nn.Flatten(start_dim=1).to(device)
        
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        embedding = self.embedding(features)
        transformer_output = self.transformer(inputs_embeds=embedding).last_hidden_state
        flattened = self.flatten(transformer_output)    
        return self.policy_net(flattened)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        embedding = self.embedding(features)
        transformer_output = self.transformer(inputs_embeds=embedding).last_hidden_state
        flattened = self.flatten(transformer_output)    
        return self.value_net(flattened)
    
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NullFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Disable orthogonal initialization
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
         self.mlp_extractor = CustomTransformerExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            hidden_dim = 32,
            num_heads = 4,
            num_layers = 4,
            max_position_embedding = 50,
            input_dim = 4)
         