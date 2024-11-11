from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, GATConv
import torch_geometric as thg
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym



class GraphFeaturesExtractor2(BaseFeaturesExtractor):

    """
    Graph feature extractor for Graph observation spaces.
    Build a graph convolutional network for belief state extraction
    
    :param observation_space:
    :param output_dim: Number of features to output from GNN. 
    """
    
    def __init__(self, observation_space: gym.spaces.Graph,
                 features_dim: int = 32, model='GCN'):
        super().__init__(observation_space, features_dim=features_dim)
        self._features_dim = features_dim
        node_feature_num = observation_space.node_space.shape[0]
        
        if model == 'GCN':
            self.conv_layer = GCNConv(node_feature_num, 2*features_dim)
        elif model == 'SAGE':
            self.conv_layer = SAGEConv(node_feature_num, 2*features_dim)
        elif model == 'GAT':
            self.conv_layer = GATConv(node_feature_num, 2*features_dim)
        
        self.conv_layer2 = GCNConv(2*features_dim, 2*features_dim) # new layer 

        self.linear_layer = nn.Linear(2*features_dim, features_dim)
    
    def forward(self, observations: thg.data.Data):
        x, edge_index, batch = observations.x, observations.edge_index, observations.batch
        h = self.conv_layer(x, edge_index).relu()
        # h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv_layer2(h, edge_index).relu()
        # h = F.dropout(h, p=0.2, training=self.training)
        # h = global_mean_pool(h, batch)


        h = self.linear_layer(h).relu()

        max_num_nodes = max(batch.bincount()).item()
        node_feature_size = h.size(1)
        num_graphs = torch.max(batch)+1
        # Initialize a tensor to hold the padded outputs
        final_output = torch.zeros((num_graphs, max_num_nodes, node_feature_size), device=h.device)

        # Fill the tensor with the node features
        for i in range(num_graphs):
            graph_mask = batch == i
            num_nodes = graph_mask.sum().item()
            final_output[i, :num_nodes, :] = h[graph_mask]
        

        node_feature_size = x.size(1)

        # Initialize a tensor to hold the padded outputs
        n_feat = torch.zeros((num_graphs, max_num_nodes, node_feature_size), device=x.device)

        # Fill the tensor with the node features
        for i in range(num_graphs):
            graph_mask = batch == i
            num_nodes = graph_mask.sum().item()
            n_feat[i, :num_nodes, :] = x[graph_mask]
        

        return final_output, (n_feat[:,:,3] != 0).long()
    
    
class GATv2(nn.Module):
    def __init__(self, observation_space, features_dim=32, hidden_size=32, heads=4, dropout_rate=0.2):
        super(GATv2, self).__init__()
        num_features = 4
        num_edge_features = 4
        
        self.conv1 = GATv2Conv(num_features, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv2 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv3 = GATv2Conv(hidden_size, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.conv4 = GATv2Conv(hidden_size, features_dim, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        
        # Additional layers for reconstruction
        self.recon_conv1 = GATv2Conv(features_dim, hidden_size, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.recon_conv2 = GATv2Conv(hidden_size, num_features, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)

    def forward(self, batch):
        x_input = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        
        x = self.conv1(x_input, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
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
            #[batch, node, hidden_dim=32]
            
        node_feature_size = x_input.size(1)

        # Initialize a tensor to hold the padded outputs
        mask = torch.zeros((batch.num_graphs, max_num_nodes, node_feature_size), device=x_input.device)

        # Fill the tensor with the node features
        for i in range(batch.num_graphs):
            graph_mask = batch.batch == i
            num_nodes = graph_mask.sum().item()
            mask[i, :num_nodes, :] = x_input[graph_mask]
            
        return final_output, (mask[:,:,3] != 0).long()


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
    def __init__(self, observation_space, features_dim=32, hidden_size=32, heads=4, dropout_rate=0.2):
        super(CustomGATv2Extractor, self).__init__(observation_space, features_dim)
        
        self.GATv2 = GATv2(observation_space, features_dim, hidden_size, heads, dropout_rate)
        # self.GATv2.load_state_dict(torch.load("model/gnn.pth", map_location=torch.device('cpu')))
        self.flatten = torch.nn.Flatten(start_dim=1)
        
    def forward(self, observations):
        output = self.GATv2(observations)
        return output