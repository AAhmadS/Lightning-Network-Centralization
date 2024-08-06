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
        hidden_dim = 16,
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
        
        self.score_activation = nn.Softmax(dim=-1).to(device)
        
        self.policy_net = nn.Linear(hidden_dim, 1).to(device)
        
        self.flatten = nn.Flatten(start_dim=1).to(device)
        
        value_net: List[nn.Module] = []
        last_layer_dim_vf = hidden_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
            
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = last_layer_dim_vf
        
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        embedding = self.embedding(features)
        transformer_output = self.transformer(inputs_embeds=embedding)
        graph_embedding = transformer_output.pooler_output #embedding of the inputs
        node_scores = self.score_activation(self.flatten(self.policy_net(transformer_output.last_hidden_state)))
        return node_scores, graph_embedding #node_scores plus the graph embedding

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        embedding = self.embedding(features)
        transformer_output = self.transformer(inputs_embeds=embedding).pooler_output
        return self.value_net(transformer_output)
    
class NullFeatureExtractor(BaseFeaturesExtractor):
    """
    acts nothing, gives the tensor of the exact shape to the transformer.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations
    
    
from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from functools import partial
import numpy as np

SelfTransformerMultiCategoricalDistribution = TypeVar("SelfTransformerMultiCategoricalDistribution", bound="TransformerMultiCategoricalDistribution")

class TransformerMultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims
        
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        self.action_dims
        action_logits = nn.Linear(latent_dim, self.action_dims[-1])
        return action_logits

    def proba_distribution(
        self: SelfTransformerMultiCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfTransformerMultiCategoricalDistribution:

        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, list(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
        

def custom_make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}
    return TransformerMultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)


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
        self.action_dist = custom_make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=self.dist_kwargs)
        self._build(lr_schedule)
        
    def _build_mlp_extractor(self) -> None:
         self.mlp_extractor = CustomTransformerExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            hidden_dim = 64,
            num_heads = 4,
            num_layers = 4,
            max_position_embedding = 50,
            input_dim = self.features_dim)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, latent_pi_score), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_score)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_pi_score: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_node = latent_pi
        mean_actions_score = self.action_net(latent_pi_score)
        mean_actions = torch.cat((action_node, mean_actions_score), dim=-1)

        if isinstance(self.action_dist, TransformerMultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:   
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, latent_pi_score), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_score)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi,latent_pi_score)
         