from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from torch.nn import LayerNorm



class GATv2(nn.Module):
    def __init__(self, observation_space, features_dim=64, hidden_size=128, heads=4, dropout_rate=0.2):
        super(GATv2, self).__init__()
        num_features = observation_space['node_features'].shape[1]
        num_edge_features = observation_space['edge_features'].shape[1]
        
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv3 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv4 = GATv2Conv(hidden_size, features_dim, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        
        self.layernorm1 = LayerNorm(hidden_size, eps=1e-6)
        self.layernorm2 = LayerNorm(hidden_size, eps=1e-6)
        self.layernorm3 = LayerNorm(hidden_size, eps=1e-6)
        
        
        self.activation_1 = nn.ELU()
        self.activation_2 = nn.ELU()
        self.activation_3 = nn.ELU()
        self.final_activation = nn.Sigmoid()
        
        # Additional layers for reconstruction
        self.recon_conv1 = GATv2Conv(features_dim, int(features_dim/2), heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.recon_conv2 = GATv2Conv(int(features_dim/2) , num_features, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)

    def forward(self, observations):
        data_list = []
        for i in range(len(observations['node_features'])):
            x = observations['node_features'][i].float()
            edge_index = observations['edge_index'][i].long()
            edge_attr = observations['edge_features'][i].float()
            data = Data(x=x.squeeze(0), edge_index=edge_index.squeeze(0), edge_attr=edge_attr.squeeze(0))
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        x_input = batch.x
        #mask tensor, passed to policy to mask the query tensors
        binary_tensor = (observations["node_features"][:, :, 3] != 0).long()
        
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        x = self.conv1(x_input, edge_index, edge_attr)
        x = self.activation_1(x)
        x = self.layernorm1(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.activation_2(x)
        x = self.layernorm2(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.activation_3(x)
        x = self.layernorm3(x)
        
        encoded_features = self.conv4(x, edge_index, edge_attr)
    
    
        # Determine the maximum number of nodes in the batch
        max_num_nodes = max(batch.batch.bincount()).item()
        node_feature_size = encoded_features.size(1)

        # Initialize a tensor to hold the padded outputs
        final_output = torch.zeros((batch.num_graphs, max_num_nodes, node_feature_size), device=encoded_features.device)

        # Fill the tensor with the node features
        for i in range(batch.num_graphs):
            graph_mask = batch.batch == i
            num_nodes = graph_mask.sum().item()
            final_output[i, :num_nodes, :] = encoded_features[graph_mask]
        
            
        return final_output, binary_tensor



class CustomGATv2Extractor(BaseFeaturesExtractor):
    """
    Implements a custom feature extractor using the GATv2 (Graph Attention Network v2) architecture.
    
    The `CustomGATv2Extractor` class inherits from `BaseFeaturesExtractor` and is responsible for extracting features from graph-structured observations. It uses two GATv2 convolutional layers to process the node features and edge attributes, and outputs a feature vector for each node in the graph.
    
    The constructor takes the following parameters:
    - `observation_space`: The observation space of the environment, which contains information about the node features and edge attributes.
    - `features_dim`: The desired dimensionality of the output feature vectors.
    - `hidden_size`: The size of the hidden layers in the GATv2 convolutions.
    - `heads`: The number of attention heads in the GATv2 convolutions.
    - `dropout_rate`: The dropout rate applied to the GATv2 convolutions.
    
    The `forward` method takes an observation dictionary as input, which contains the node features, edge indices, and edge attributes. It applies the two GATv2 convolutions to the input, and returns a tensor of output feature vectors, where each row corresponds to a node in the graph.
    """
    def __init__(self, observation_space, features_dim=32, hidden_size=128, heads=4, dropout_rate=0.2):
        super(CustomGATv2Extractor, self).__init__(observation_space, features_dim)
        
        self.GATv2 = GATv2(observation_space, int(features_dim), hidden_size, heads, dropout_rate)
        self.GATv2.load_state_dict(torch.load("plotting/gnn.pth", map_location=torch.device('cpu')))
        # for param in self.GATv2.parameters():
        #     param.requires_grad = False
        self.flatten = torch.nn.Flatten(start_dim=1)
        
    def forward(self, observations):
        output, mask = self.GATv2(observations)
        return (output, mask)