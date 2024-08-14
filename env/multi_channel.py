import gym
from gym import spaces
from gym.spaces import *
from gym.utils import seeding
import numpy as np
from simulator import preprocessing
from simulator.simulator import simulator
from simulator.preprocessing import generate_transaction_types
import time

import random
from collections import Counter
import networkx as nx


from scipy.special import softmax
import math


class FeeEnv(gym.Env):
    """
    ### Description

    This environment corresponds to the LIGHTNING NETWORK simulation. A source node is chosen and a local network
    around that node with radius 2 is created and at each time step, a certain number of transitions are being simulated.

    ### Scales

    We are using the following scales for simulating the real world Lightning Network:

    - Fee Rate: msat                                      - Base Fee: msat
    - Transaction amounts: sat                            - Reward(income): msat
    - Capacity: sat                                       - Balance: sat

    ### Action Space

    The action is a `ndarray` with shape `(2*n_channel,)` which can take values `[0,upper bound]`
    indicating the fee rate and base fee of each channel starting from source node.

    | dim       | action                 | dim        | action                |
    |-----------|------------------------|------------|-----------------------|
    | 0         | fee rate channel 0     | 0+n_channel| fee base channel 0    |
    | ...       |        ...             | ...        |         ...           |
    | n_channel | fee rate last channel  | 2*n_channel| fee base last channel |

    ### Observation Space

    The observation is a `ndarray` with shape `(2*n_channel,)` with the values corresponding to the balance of each
    channel and also accumulative transaction amounts in each time steps.

    | dim       | observation            | dim        | observation                 |
    |-----------|------------------------|------------|-----------------------------|
    | 0         | balance channel 0      | 0+n_channel| sum trn amount channel 0    |
    | ...       |          ...           | ...        |            ...              |
    | n_channel | balance last channel   | 2*n_channel| sum trn amount last channel |

    ### Rewards

    Since the goal is to maximize the return in the long term, the reward is the sum of incomes from fee payments of each channel.
    The reward scale is Sat to control the upper bound.

    ***Note:
    We are adding the income from each payment to the balance of the corresponding channel.
    """

    def __init__(self, data, max_capacity, max_episode_length, number_of_transaction_types, counts,
                  amounts, epsilons, capacity_upper_scale_bound, LN_graph, seed):
        
        self.max_capacity = max_capacity
        self.capacity_upper_scale_bound = capacity_upper_scale_bound
        self.data = data
        self.LN_graph = LN_graph
        self.max_episode_length = max_episode_length
        
        # self.seed = seed
        self.src = self.data['src']
        self.providers = data['providers']
        self.local_heads_number = data['local_heads_number']
        self.n_channel = data['n_channels']
        self.prev_reward = 0
        self.total_time_step = 0
        self.time_step = 0
        self.prev_action = [] 

        self.undirected_attributed_LN_graph = self.set_undirected_attributed_LN_graph()
        self.transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts, epsilons)

        self.set_new_graph_environment()

        self.n_nodes = len(self.data['nodes'])


        
        #Action Space
        self.action_space = spaces.MultiDiscrete([self.n_nodes, self.capacity_upper_scale_bound])

        self.num_node_features = len(next(iter(self.simulator.current_graph.nodes(data=True)))[1]['feature'])
        self.num_edge_features = len(next(iter(self.simulator.current_graph.edges(data=True)))[2]) -1

        
        #Observation Space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_nodes, self.num_node_features), dtype=np.float32)


        # node_features = self.extract_graph_attributes(self.simulator.current_graph, [], exclude_attributes=['capacity', 'channel_id'])

        # self.state = node_features
        
        print("num_node_features:", self.num_node_features)
        print("num_edge_features:", self.num_edge_features)

        print("number of nodes: ",self.n_nodes)

        random.seed(44)


        num_edges = len(self.simulator.current_graph.edges())

        self.node_features_space = spaces.Box(low=0, high=1, shape=(self.n_nodes, self.num_node_features), dtype=np.float32)
        self.edge_features_space = spaces.Box(low=0, high=1, shape=(num_edges, self.num_edge_features), dtype=np.float32)
        self.edge_index_space = spaces.Box(low=0, high=self.n_nodes, shape=(2, num_edges), dtype=np.float32)

        self.observation_space = self.node_features_space
        # self.observation_space = spaces.Dict({
        #     "node_features" : self.node_features_space,
        #     "edge_features" : self.edge_features_space,
        #     "edge_index": self.edge_index_space
        # })
        
        node_features, edge_index, edge_features = self.extract_graph_attributes(self.simulator.current_graph,["capacity", "id"])

        # self.state = {
        #     "node_features" : node_features,
        #     "edge_features": edge_features,
        #     "edge_index": edge_index
        # }
        self.state = node_features

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def step(self, action):
        
        # action = self.aggregate_and_standardize_action(action)
        
        if self.total_time_step % 500 == 0:
            print("action: ",action,"time step: ",self.time_step)

        
        
        new_trg = self.graph_nodes[action[0]]
        if new_trg not in self.simulator.trgs:
            self.simulator.trgs.append(new_trg)
            # self.simulator.shares.append(action[1])
            self.simulator.shares[new_trg] = action[1] + 1
        else:
            budget_so_far = self.simulator.shares[new_trg]
            self.simulator.shares[new_trg] = budget_so_far + action[1] + 1



        action = self.map_action_to_capacity()
        

        additive_channels, ommitive_channels = self.simulator.update_network_and_active_channels(action, self.prev_action)

        self.prev_action = action
        
        additive_channels_fees = self.simulator.get_channel_fees(additive_channels)
        
        self.simulator.update_amount_graph(additive_channels, ommitive_channels, additive_channels_fees)


        fees = self.simulator.get_channel_fees(self.simulator.trgs + self.simulator.trgs)


        _, transaction_amounts, transaction_numbers = self.simulate_transactions(fees, self.simulator.trgs)

        # if self.time_step == self.max_episode_length - 1: 
        reward = 1e-6*(np.sum(np.multiply(self.simulator.src_fee_rate, transaction_amounts ) + \
                np.multiply(self.simulator.src_fee_base, transaction_numbers)))
        # else: 
        #     reward = 0
        

        
        reward = reward - self.prev_reward
        self.prev_reward += reward


        self.time_step += 1
        self.total_time_step += 1

        

        

        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}

        done = self.time_step >= self.max_episode_length

        # capacities_list = np.zeros((self.n_nodes,))
        
      
        # self.simulator.current_graph = self.evolve_graph()

        node_features, edge_index, edge_features = self.extract_graph_attributes(self.simulator.current_graph,["capacity", "id"])
        # node_features = self.extract_graph_attributes(self.simulator.current_graph, transaction_amounts, exclude_attributes=['capacity', 'channel_id'])
        self.state = node_features

                



        # self.state = {

        # "node_features" : node_features,
        # "edge_features" : edge_features,
        # "edge_index": edge_index

        # }
        


        return self.state, reward, done, info
    
    def generate_number_of_new_channels(self, time_step):
        #TODO: generate the number of added channels base on time step
        return 7

    def simulate_transactions(self, fees, trgs):
        
        #NOTE: fees set in the step, now will be added to network_dict and active_channels
        self.simulator.set_channels_fees(fees, trgs)

        output_transactions_dict = self.simulator.run_simulation()

        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers
    

    def reset(self):
        
        self.time_step = 0
        self.prev_action = []
        self.prev_reward = 0
        self.set_new_graph_environment()

        # self.remaining_capacity = self.max_capacity

        node_features, edge_index, edge_features = self.extract_graph_attributes(self.simulator.current_graph,["capacity", "id"])
        # self.state = {
        # "node_features" : node_features,
        # "edge_features" : edge_features,
        # "edge_index": edge_index
        # }
        # node_features = self.extract_graph_attributes(self.simulator.current_graph, [], exclude_attributes=['capacity', 'channel_id'])
        self.state = node_features

        return self.state 


    def action_fix_index_to_capacity(self,capacities,action):
        """
        Fixes the index values in an action list to match the corresponding capacity values.
        
        Args:
            capacities (list): A list of capacity values.
            action (list): A list of graph node indices.
        
        Returns:
            list: A new list with the graph node indices in the first half and the corresponding capacity values in the second half.
        """
        midpoint = len(action) // 2
        fixed_action = [self.graph_nodes[i] for i in action[:midpoint]]
        fixed_action.extend([capacities[i] for i in action[midpoint:]])
        return fixed_action
    
    def map_action_to_capacity(self):
        """
        Maps an action to a list of target nodes and their corresponding capacities.
        
        The action is assumed to be a list where the first half represents the indices of the target nodes, and the second half represents the capacities for those targets.
        
        Args:
            action (list): A list containing the indices of the target nodes and their corresponding capacities.
        
        Returns:
            list: A list containing the target nodes and their corresponding capacities.
        """
        # midpoint = len(action) // 2
        # fixed_action = []
        #setting up trgs from their ind
        # fixed_trgs = [self.graph_nodes[i] for i in action[:midpoint]]


        # fixed_trgs = [self.graph_nodes[action[0]]]
        # fixed_action = [action[1] * self.remaining_capacity / self.capacity_upper_scale_bound]
        # self.remaining_capacity = self.remaining_capacity - fixed_action[0]

        # fixed_trgs = [self.graph_nodes[action]]
        # fixed_action = [self.max_capacity / self.max_episode_length]

        # fixed_trgs = self.simulator.trgs
        # fixed_action = list((np.array(self.simulator.shares)/self.max_episode_length) * self.max_capacity)
        shares_list = list(self.simulator.shares.values())
        trgs_list = list(self.simulator.shares.keys())
        shares_sum = sum(shares_list) 
        caps = [item / shares_sum * self.max_capacity for item in shares_list]
        trgs_and_caps = trgs_list + caps
        

        # if len(action) != 0:
        #     fixed_action = list(softmax(np.array(action[midpoint:])) * self.maximum_capacity)    
      
        return trgs_and_caps
    
        
        return nonzero_unique_nodes + action_bal
    

    def get_local_graph(self, scale):
        return self.simulator.get_local_graph(scale)
        # return self.simulator.current_graph
    
    def set_undirected_attributed_LN_graph(self):
        """    
        Sets the undirected attributed Lightning Network (LN) graph for the environment.
        
        Returns:
            networkx.Graph: The undirected attributed LN graph.
        """
        undirected_G = nx.Graph(self.LN_graph)
        return undirected_G

    def sample_graph_environment(self, local_size):
        random.seed(44)
        sampled_sub_nodes = preprocessing.fireforest_sample(self.undirected_attributed_LN_graph, local_size, providers=self.providers, local_heads_number=self.local_heads_number)    
        return sampled_sub_nodes
    
    def evolve_graph(self):
        """
        Generates the number of new channels to create for the current time step.
        
        Returns:
            int: The number of new channels to create.
        """
        number_of_new_channels = self.generate_number_of_new_channels(self.time_step)

        transformed_graph = self.add_edges(self.simulator.current_graph, number_of_new_channels)

        return transformed_graph
    
    def fetch_new_pairs_for_create_new_channels(self, G, number_of_new_channels):
        """
        Fetches a list of (source, target) pairs for creating new channels in the network.
        
        The function generates a list of pairs based on the logarithmic degree distribution and the inverse logarithmic degree distribution of the nodes in the network. The number of pairs returned is equal to the `number_of_new_channels` parameter.
        
        Args:
            G (networkx.Graph): The network graph.
            number_of_new_channels (int): The number of new channels to create.
        
        Returns:
            list of (str, str): A list of (source, target) pairs for the new channels.
        """
        #Return a list of tuples containing (src,trg) pairs for each channel to be created.
        #[(src1,trg1), (src2,trg2),...]
        list_of_pairs = []
        degree_sequence = [d for n, d in G.degree()]

        # Create distribution based on logarithm of degree
        log_degree_sequence = np.log(degree_sequence)
        log_degree_distribution = {node: deg for node, deg in zip(G.nodes(), log_degree_sequence)}

        # Create distribution based on inverse of the logarithmic degree
        inv_log_degree_sequence = 1 / log_degree_sequence
        inv_log_degree_distribution = {node: deg for node, deg in zip(G.nodes(), inv_log_degree_sequence)}
        # random.seed(self.time_step + 42)
        for i in range(number_of_new_channels):
            trg = random.choices(list(log_degree_distribution.keys()),
                                  weights=log_degree_distribution.values(), k=1)[0]
            src = random.choices(list(inv_log_degree_distribution.keys()),
                                  weights=inv_log_degree_distribution.values(), k=1)[0]
            if trg == src:
                continue
            list_of_pairs.append((src, trg))

        return list_of_pairs

    def add_edges(self, G, k): 

        list_of_pairs = self.fetch_new_pairs_for_create_new_channels(G, k)

        fees = self.simulator.get_rates_and_bases(list_of_pairs)

        list_of_balances = self.simulator.update_evolved_graph(fees, list_of_pairs)

        midpoint = len(fees) // 2

        for ((src,trg), bal, fee_base_src, fee_base_trg, fee_rate_src, fee_rate_trg) in zip(list_of_pairs, 
                                                                                            list_of_balances,
                                                                                            fees[midpoint:][1::2], 
                                                                                            fees[midpoint:][::2], 
                                                                                            fees[:midpoint][1::2], 
                                                                                            fees[:midpoint][::2]):            
            # Add edge if not already exists
            if not G.has_edge(src, trg):
                G.add_edge(src, trg, capacity = 2*bal, fee_base_msat = fee_base_src , fee_rate_milli_msat = fee_rate_src , balance = bal)
                G.add_edge(trg, src, capacity = 2*bal, fee_base_msat = fee_base_trg , fee_rate_milli_msat = fee_rate_trg, balance = bal) 
                self.simulator.evolve_network_dict(src, trg, fee_base_src, fee_rate_src,fee_base_trg,fee_rate_trg, bal)

        return G
    
    def set_new_graph_environment(self):

        sub_nodes = self.sample_graph_environment(local_size = self.data["local_size"])
        
        network_dictionary, sub_providers, sub_edges, sub_graph = preprocessing.get_sub_graph_properties(self.LN_graph, sub_nodes, self.providers)
        
        node_variables, active_providers, _ = preprocessing.init_node_params(sub_edges, sub_providers, verbose=False)
       
    
        self.data['network_dictionary'] = network_dictionary
        self.data['node_variables'] = node_variables
        self.data["capacity_max"] = max(node_variables["total_capacity"])
        self.data['active_providers'] = active_providers
        self.data['nodes'] = sub_nodes
        
        self.graph_nodes = sub_nodes
        

        self.simulator = simulator(
                                   src=self.src,
                                   network_dictionary=self.data['network_dictionary'],
                                   merchants = self.providers,
                                   transaction_types=self.transaction_types,
                                   node_variables=self.data['node_variables'],
                                   active_providers=self.data['active_providers'],
                                   fee_policy = self.data["fee_policy"],
                                   fixed_transactions=False,
                                   graph_nodes = self.graph_nodes,
                                   current_graph = sub_graph)
        
        
        
    def extract_graph_attributes(self, G, exclude_attributes=None):

        """
        Extracts node features, edge indices, and edge attributes from a given graph `G`.

        Args:
            G (networkx.Graph): The input graph.
            exclude_attributes (list or None): List of attribute names to exclude (optional).

        Returns:
            tuple:
                - node_features (numpy.ndarray): A 2D array of node features.
                - edge_index (numpy.ndarray): A 2D array of edge indices.
                - edge_attr (numpy.ndarray): A 2D array of edge attributes.
        """
        
        # node_features = np.array([G.nodes[n]['feature'] for n in self.graph_nodes]).astype(np.float32)
        node_features = np.zeros(shape = (self.n_nodes, self.num_node_features))
        nodes_list = G.nodes(data = True)

        degrees = preprocessing.get_nodes_degree_centrality(self.simulator.current_graph)

        if np.max(self.simulator.nodes_cumulative_trs_amounts) == 0:
            normalized_transaction_amounts = np.zeros_like(self.simulator.nodes_cumulative_trs_amounts)
        else:
            normalized_transaction_amounts = self.simulator.nodes_cumulative_trs_amounts / np.sum(self.simulator.nodes_cumulative_trs_amounts)
            
        
        #set node features 
        for node in nodes_list:
            node_features[self.simulator.map_nodes_to_id[node[0]]][0] = degrees[node[0]]
            node_features[self.simulator.map_nodes_to_id[node[0]]][1] = G.nodes[node[0]]["feature"][1]
            node_features[self.simulator.map_nodes_to_id[node[0]]][2] = normalized_transaction_amounts[self.simulator.map_nodes_to_id[node[0]]]
            node_features[self.simulator.map_nodes_to_id[node[0]]][3] = 0
            if node[0] in self.simulator.trgs:
                node_features[self.simulator.map_nodes_to_id[node[0]]][3] = self.simulator.shares[node[0]]/sum(self.simulator.shares.values())
        


        # if self.time_step == 1 :
        #     for e in G.edges(data=True):
        #         node_features[self.simulator.map_nodes_to_id[e[0]]][3] = 0



        # for e in G.edges(data=True):
        #     node_features[self.simulator.map_nodes_to_id[e[0]]][3] += e[2]['capacity'] / 2

        # max_list = self.get_normalizer_configs()
        # max_total_budget = max(node_features[:,3])



        # for i in range(len(self.graph_nodes)):
        #     node_features[i][0] = degrees[nodes_list[i]]
        #     # node_features[i][1] = eigenvectors[nodes_list[i]]
        #     node_features[i][2] = 0
        #     if i in trgs:
        #         # node_features[i][4] = 
        #         # print("Target node : ",nodes_list[i])
        #         # print("trgs:", self.simulator.trgs)
        #         node_features[i][2] = self.simulator.network_dictionary[(self.src,nodes_list[i])][0] / self.max_capacity

                # node_features[i][4] = transaction_amounts[trgs.index(i)] / max_list[3]
            # node_features[i][5] = normalized_transaction_amounts[i]

            # node_features[i][5] = node_features[i][5]/max_list[0]
            # node_features[i][6] = node_features[i][6]/max_list[1]
            # node_features[i][3] = node_features[i][3] / max_total_budget
      


            
        
        
        # Extract edge index
        edge_index = np.array([(self.simulator.map_nodes_to_id[x], self.simulator.map_nodes_to_id[y]) for (x,y) in G.edges]).T


        # Extract multiple edge attributes (excluding specified attributes)
        max_list = self.get_normalizer_configs()
        edge_attr_list = []
        for e in G.edges(data=True):
            filtered_attrs = {key: e[2][key] for key in e[2] if key not in exclude_attributes}
            filtered_attrs = list(filtered_attrs.values())
            edge_attr_list.append([filtered_attrs[i]/max_list[i] for i in range(len(max_list))])
        edge_attr = np.array(edge_attr_list)

        # self.compare_and_update(edge_attr)
        return node_features, edge_index, edge_attr
        # return node_features, edge_index



        return node_features

    def get_normalizer_configs(self):
        #return cap_max, base_max, rate_max
        return self.data["fee_base_max"], self.data["fee_rate_max"], self.data["capacity_max"], 200*(10000+50000+100000) # maximum amount of transaction per step
    