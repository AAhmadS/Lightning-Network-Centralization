import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import numpy as np
from tqdm import tqdm 
import os
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
        self.activation_4 = nn.ELU()
        self.final_activation = nn.Sigmoid()
        
        # Additional layers for reconstruction
        self.recon_conv1 = GATv2Conv(features_dim, int(features_dim/2), heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)
        self.recon_conv2 = GATv2Conv(int(features_dim/2) , num_features, heads=heads, edge_dim=num_edge_features, dropout=dropout_rate, concat=False)

    def forward(self, observations):
        data_list = []
        for i in range(len(observations['node_features'])):
            x = observations['node_features'][i].clone().detach().float()
            edge_index = observations['edge_index'][i].clone().detach().long()
            edge_attr = observations['edge_features'][i].clone().detach().float()
            data = Data(x=x.squeeze(0), edge_index=edge_index.squeeze(0), edge_attr=edge_attr.squeeze(0))
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        x_input = batch.x
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
        
        recon_x = self.recon_conv1(encoded_features, edge_index, edge_attr)
        recon_x = self.activation_4(recon_x)
        recon_x = self.recon_conv2(recon_x, edge_index, edge_attr)
        recon_x = self.final_activation(recon_x)
        
            
        return x_input, recon_x 

class GNN_Agent:
    
    def __init__(self, env, device, features_dim):
        self.batch_size = 64
        self.env = env
        self.device = device
        self.features_dim = features_dim
        self.observation_space = env.observation_space
        self.model = GATv2(observation_space=self.observation_space, features_dim=features_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
    def sample_observations(self):
        observations = {"node_features": [], "edge_features": [], "edge_index": []}
        done = True
        
        while len(observations["node_features"]) < self.batch_size:
            if done:
                state = self.env.reset()
            action = [np.random.randint(0, 50), np.random.randint(0, 10)]  # Random action with the given dimensions
            state, reward, done, info = self.env.step(action)
            
            observations["node_features"].append(state["node_features"])
            observations["edge_features"].append(state["edge_features"])
            observations["edge_index"].append(state["edge_index"])
        
        # Convert lists to numpy arrays
        for key in observations:
            observations[key] = np.array(observations[key])
        
        return observations
    
    def train(self, total_timesteps):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(total_timesteps/self.batch_size), eta_min=1e-5)

        timestep = 0
        best_loss = 1
        # Wrap the range with tqdm for a progress bar
        with tqdm(total=total_timesteps, desc="Training") as pbar:
            while timestep < total_timesteps:
                # Sample a batch of observations
                observations = self.sample_observations()

                # Convert the observations to torch tensors
                for key in observations:
                    observations[key] = torch.tensor(observations[key], dtype=torch.float if key != "edge_index" else torch.long, device=self.device)

                # Forward pass through the GNN model
                base_x, recon_x = self.model(observations)

                # Compute the loss
                recon_loss = self.criterion(recon_x, base_x)

                # Backward pass and optimization step
                self.optimizer.zero_grad()
                recon_loss.backward()
                self.optimizer.step()

                # Step the scheduler
                self.scheduler.step()

                timestep += self.batch_size  
                pbar.update(self.batch_size)  
                pbar.set_postfix(loss=recon_loss.item(), lr=self.scheduler.get_last_lr()[0]) 
                
                if recon_loss.item() < best_loss:
                    print("saving the model, new best_loss: ", recon_loss.item())
                    best_loss = recon_loss.item()
                    self.save_model(os.path.join("plotting","gnn.pth"))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
