import gym
from gym import spaces
from gym.spaces import *
from gym.utils import seeding
import numpy as np
import graph_embedding_processing
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

    def __init__(self, mode, data, max_capacity, fee_base_upper_bound, max_episode_length,
                  number_of_transaction_types, counts, amounts, epsilons, capacity_upper_scale_bound,
                    seed, LN_graph):
        # Source node\
        self.total_time_step = 0
        self.data = data
        self.src = self.data['src']
        self.LN_graph = LN_graph
        self.undirected_attributed_LN_graph, self.reverse_mapping = self.set_undirected_attributed_LN_graph()
        self.providers = data['providers']
        self.mode = mode
        self.transaction_types = generate_transaction_types(number_of_transaction_types, counts, amounts,
                                                       epsilons)
        self.set_new_graph_environment()
        self.n_nodes = len(self.data['nodes'])
        
        
        self.prev_action = []
        
        
        self.n_channel = data['n_channels']
        
        
        
         # nodes should be minus one to not include our node
       
        print("number of nodes: ",self.n_nodes)
        self.graph_nodes = list(self.data['nodes'])
        if self.src in self.graph_nodes:
            self.graph_nodes.remove(self.src)
        
        print('action dim:', self.n_nodes)
        

        #NOTE: The following lines are for fee selection mode
        '''# Base fee and fee rate for each channel of src
        self.action_space = spaces.Box(low=-1, high=+1, shape=(2 * self.n_channel,), dtype=np.float32)
        self.fee_rate_upper_bound = 1000
        self.fee_base_upper_bound = fee_base_upper_bound

        # Balance and transaction amount of each channel
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * self.n_channel,), dtype=np.float32)

        # Defining action space & editing of step function in multi-channel
        # first n_channels are id's  of connected nodes and the seconds are corresponding  capacities
        # add self.capacities to the fields of env class'''

        self.maximum_capacity = max_capacity

        self.action_space = spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_channel)] + [capacity_upper_scale_bound for _ in range(self.n_channel)])
        

        
        num_node_features = len(next(iter(self.simulator.current_graph.nodes(data=True)))[1]['feature'])
        num_edge_features = len(next(iter(self.simulator.current_graph.edges(data=True)))[2]) - 2

        print("num_node_features:",num_node_features)
        print("num_edge_features:",num_edge_features)

        num_edges = len(self.simulator.current_graph.edges())
        self.node_features_space = spaces.Box(low=0, high=1, shape=(self.n_nodes, num_node_features), dtype=np.float32)
        self.edge_features_space = spaces.Box(low=0, high=1, shape=(num_edges, num_edge_features), dtype=np.float32)
        self.edge_index_space = spaces.Box(low=0, high=self.n_nodes, shape=(2, num_edges), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "node_features" : self.node_features_space,
            "edge_attr" : self.edge_features_space,
            "edge_index": self.edge_index_space
        })


        # self.observation_space = Dict({
        #     'capacities': Box(low=0, high=capacity_upper_scale_bound, shape=(self.n_nodes,)),
        #     'transaction_amounts': Box(low=0, high=np.inf, shape=(self.n_nodes,)),
        #     'graph_embedding': Box(low=-np.inf, high=np.inf, shape=(self.embedding_size,))
        # })

        #NOTE: Initial values of each channel for fee selection mode
        # self.initial_balances = data['initial_balances']
        # self.capacities = data['capacities']
        # self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
        

        # self.graph_embedding =self.get_new_graph_embedding(self.simulator.current_graph,self.embedding_mode)

        # self.state = {
        #     'capacities': np.zeros(self.n_nodes),
        #     'transaction_amounts': np.zeros(self.n_nodes),
        #     'graph_embedding': self.graph_embedding
        # }
        
        node_features, edge_index, edge_attr = self.extract_graph_attributes(self.simulator.current_graph, exclude_attributes=['capacity', 'channel_id'])
        
        self.state = {

            "node_features" : node_features,
            "edge_attr" : edge_attr,
            "edge_index": edge_index
        }
            
        self.time_step = 0
        self.max_episode_length = max_episode_length
        
        self.transaction_amounts_list = np.zeros((self.n_nodes,))

        
        #NOTE: for fee selection
        # self.balance_ratio = 0.1

        # Simulator
        
        # self.simulator = simulator(mode=self.mode,
        #                            src=self.src,
        #                            trgs=self.data['trgs'],
        #                            channel_ids=self.data['channel_ids'],
        #                            active_channels=self.data['active_channels'],
        #                            network_dictionary=self.data['network_dictionary'],
        #                            merchants=self.providers,
        #                            transaction_types=self.transaction_types,
        #                            node_variables=self.data['node_variables'],
        #                            active_providers=self.data['active_providers'],
        #                            fee_policy = self.data["fee_policy"],
        #                            fixed_transactions=False)
        # self.seed(seed)
        
        



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #NOTE: might need to add fees attr to set_channel_fees in the following
    # def step(self, action, rescale=True):
    #     # Rescaling the action vector (fee selection mode)
    #     # if rescale:
    #     #     action[0:self.n_channel] = .5 * self.fee_rate_upper_bound * action[0:self.n_channel] + \
    #     #                                .5 * self.fee_rate_upper_bound
    #     #     action[self.n_channel:2 * self.n_channel] = .5 * self.fee_base_upper_bound * action[
    #     #                                                                                  self.n_channel:2 * self.n_channel] + \
    #     #                                                 .5 * self.fee_base_upper_bound

    #     # Running simulator for a certain time interval
    #     balances, transaction_amounts, transaction_numbers = self.simulate_transactions(action)
    #     self.time_step += 1

    #     reward = 1e-6 * np.sum(np.multiply(action[0:self.n_channel], transaction_amounts) + \
    #                     np.multiply(action[self.n_channel:2 * self.n_channel], transaction_numbers))

    #     info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}

    #     done = self.time_step >= self.max_episode_length

    #     self.state = np.append(balances, transaction_amounts)/1000

    #     return self.state, reward, done, info
    
    
    def step(self, action):
        
        action = self.aggregate_and_standardize_action(action)
        
        if self.total_time_step % 500==0:
            print("action: ",action,"time step: ",self.time_step)
  
    
        action = self.map_action_to_capacity(action)
        midpoint = len(action) // 2

        self.simulator.trgs = action[:midpoint]
        self.n_channel = midpoint
         
        '''
        In the following couple of lines, new channels are being added to the network, along with active_
        _channels dict and also, channels not present anymore, will be deleted
        '''
        #attention: budget removed
        additive_channels, ommitive_channels = self.simulator.update_network_and_active_channels(action, self.prev_action)

        self.prev_action = action

        fees = self.simulator.get_additive_channel_fees(action)
        
        self.simulator.update_amount_graph(additive_channels, ommitive_channels,fees)

        fees_to_use_for_reward = fees[::2]
        # if self.time_step + 1 == self.max_episode_length :
        #     reward = 0
            
        #     for i in range(self.max_episode_length):
        #         balances, transaction_amounts, transaction_numbers = self.simulate_transactions(fees,additive_channels)
        #         reward += 1e-6 *(np.sum(np.multiply(fees_to_use_for_reward[0:self.n_channel], transaction_amounts) + \
        #                 np.multiply(fees_to_use_for_reward[self.n_channel:], transaction_numbers)))

        # else:
        balances, transaction_amounts, transaction_numbers = self.simulate_transactions(fees,additive_channels)
        reward = 1e-6 *(np.sum(np.multiply(fees_to_use_for_reward[0:self.n_channel], transaction_amounts ) + \
                np.multiply(fees_to_use_for_reward[self.n_channel:], transaction_numbers)))

        self.time_step += 1
        self.total_time_step+=1
        info = {'TimeLimit.truncated': True if self.time_step >= self.max_episode_length else False}
        done = self.time_step >= self.max_episode_length

        # capacities_list = np.zeros((self.n_nodes,))
        
        # for idx, in range(midpoint):
        #     capacities_list[raw_action[idx]] = raw_action[idx+midpoint]
        #     self.transaction_amounts_list[raw_action[idx]] += transaction_amounts[idx]       
        # self.state = {
        #     'capacities': capacities_list,
        #     'transaction_amounts': self.transaction_amounts_list,
        #     'graph_embedding': self.graph_embedding
        # } 
        #TODO: use balances and transaction amounts here for edeg and node attributes, calculate the centralities here. 
        self.simulator.current_graph = self.evolve_graph()

        node_features, edge_index, edge_attr = self.extract_graph_attributes(self.simulator.current_graph, exclude_attributes=['capacity', 'channel_id'])

        self.state = {

        "node_features" : node_features,
        "edge_attr" : edge_attr,
        "edge_index": edge_index

        }
        


        return self.state, reward, done, info
    
    def generate_number_of_new_channels(self, time_step):
        #TODO: generate the number of added channels base on time step
        return 7

    def simulate_transactions(self, action, additive_channels = None):
        """
        Simulates transactions for the given action and additive channels.
        
        Args:
            action (str): The action to simulate.
            additive_channels (list, optional): A list of additive channels to include in the simulation.
        
        Returns:
            tuple: A tuple containing the balances, transaction amounts, and transaction numbers resulting from the simulation.
        """
        #NOTE: fees set in the step, now will be added to network_dict and active_channels
        self.simulator.set_channels_fees(self.mode,action,additive_channels[:len(additive_channels)//2])

        output_transactions_dict = self.simulator.run_simulation(action)
        balances, transaction_amounts, transaction_numbers = self.simulator.get_simulation_results(action,
                                                                                                   output_transactions_dict)

        return balances, transaction_amounts, transaction_numbers
    

    def reset(self):
        # print('episode ended!')
        self.time_step = 0
        if self.mode == 'fee_setting':
            self.state = np.append(self.initial_balances, np.zeros(shape=(self.n_channel,)))
            return np.array(self.state, dtype=np.float64)
        
        else:
            self.prev_action = []
            self.set_new_graph_environment()
            # self.graph_embedding = self.get_new_graph_embedding(self.simulator.current_graph,self.embedding_mode)
            # self.state = {
            #     'capacities': np.zeros(self.n_nodes),
            #     'transaction_amounts': np.zeros(self.n_nodes),
            #     'graph_embedding': self.graph_embedding #sample new embedding
            # }
            node_features, edge_index, edge_attr = self.extract_graph_attributes(self.simulator.current_graph, exclude_attributes=['capacity', 'channel_id'])
            self.state = {
            "node_features" : node_features,
            "edge_attr" : edge_attr,
            "edge_index": edge_index
            }
            self.transaction_amounts_list = np.zeros((self.n_nodes,))
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
    
    def map_action_to_capacity(self, action):
        """
        Maps an action to a list of target nodes and their corresponding capacities.
        
        The action is assumed to be a list where the first half represents the indices of the target nodes, and the second half represents the capacities for those targets.
        
        Args:
            action (list): A list containing the indices of the target nodes and their corresponding capacities.
        
        Returns:
            list: A list containing the target nodes and their corresponding capacities.
        """
        midpoint = len(action) // 2
        fixed_action = []
        #seeting up trgs from their ind
        fixed_trgs = [self.graph_nodes[i] for i in action[:midpoint]]
        
        if len(action) != 0:
            fixed_action = list(softmax(np.array(action[midpoint:])) * self.maximum_capacity)    
      
        return fixed_trgs+fixed_action
    
    def aggregate_and_standardize_action(self,action):
        """
        Aggregates and standardizes the action values in the given action list.
        
        The action list is assumed to be a concatenation of node IDs and their corresponding action values. This function first identifies the unique nodes, then aggregates the action values for each node, and finally standardizes the aggregated action values by finding the greatest common divisor (GCD) of the values and dividing each value by the GCD.
        
        Args:
            action (list): A list containing node IDs and their corresponding action values.
        
        Returns:
            list: A list containing the unique nodes and their standardized action values.
        """
        midpoint = len(action) // 2
        unique_nodes = list(set(action[:midpoint]))
        nonzero_unique_nodes = []
        action_bal = []
          
        for node in unique_nodes:
            agg_bal = 0
            for i in range(midpoint):
                if action[i] == node:
                    agg_bal += action[i+midpoint]
            if agg_bal !=0:
                nonzero_unique_nodes.append(node)
                action_bal.append(agg_bal)
        
        #Standardizing the balances
        bal_gcd = math.gcd(*action_bal)
        action_bal = [balance/bal_gcd for balance in action_bal]
        
        return nonzero_unique_nodes + action_bal
    
    def action_fix(action):
        """
        Extracts the connected node IDs and their corresponding capacities from an action.
        
        Args:
            action (list): A list of values representing the capacities of connected nodes.
        
        Returns:
            list: A list containing the connected node IDs and their corresponding capacities.
        """
        connected_node_ids = []
        connected_node_capacities = []
        for i, val in enumerate(action):
            if val != 0:
                connected_node_ids.append(i)
                connected_node_capacities.append(val)
        return connected_node_ids + connected_node_capacities
            


<<<<<<< HEAD
    def get_local_graph(self,scale):
=======
    def get_local_graph(self, scale):
>>>>>>> e55c3bd3ec582db156c6cc49dffa4ec54f2acc37
        return self.simulator.get_local_graph(scale)
        # return self.simulator.current_graph
    
    def set_undirected_attributed_LN_graph(self):
        undirected_G = nx.Graph(self.LN_graph)
        # Reindex nodes to be numeric
        mapping = {node: i for i, node in enumerate(undirected_G.nodes())}
        numeric_undirected_G = nx.relabel_nodes(undirected_G, mapping)
        # Adding node attributes to the numeric graph
        for node, data in undirected_G.nodes(data=True):
            numeric_undirected_G.nodes[mapping[node]].update(data)

        reverse_mapping = {i: node for node, i in mapping.items()}
        return numeric_undirected_G, reverse_mapping

        
    

    def sample_graph_environment(self):
        #TODO: this should not be set manually and should get pramaeter but can't use self.data['nodes']
        random.seed()
        sampled_sub_nodes = preprocessing.get_fire_forest_sample(self.undirected_attributed_LN_graph, self.reverse_mapping, 100)    
        return sampled_sub_nodes
    
    def evolve_graph(self):
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
        random.seed()
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

        sub_nodes = self.sample_graph_environment()

        network_dictionary, sub_providers, sub_edges, sub_graph = preprocessing.get_sub_graph_properties(self.LN_graph,sub_nodes,self.providers)

        # adding these features in order: degree_centrality, eigenvectors_centrality, is_provider, is_connected_to_us, normalized_transaction_amount]
        
        # sub_graph = self.make_graph_weighted(sub_graph, amount = self.average_transaction_amounts)

        active_channels = preprocessing.create_active_channels(network_dictionary, [])

        try:
            node_variables, active_providers, _ = preprocessing.init_node_params(sub_edges, sub_providers, verbose=False)
        except:
            node_variables, active_providers = None
            


        self.data['active_channels'] = active_channels
        self.data['network_dictionary'] = network_dictionary
        self.data['node_variables'] = node_variables
        self.data['active_providers'] = active_providers
        self.data['nodes'] = sub_graph.nodes()
        

        

        self.graph_nodes = list(self.data['nodes'])
        if self.src in self.graph_nodes:
            self.graph_nodes.remove(self.src)



        self.simulator = simulator(mode=self.mode,
                                   src=self.src,
                                   trgs=self.data['trgs'],
                                   channel_ids=self.data['channel_ids'],
                                   active_channels=self.data['active_channels'],
                                   network_dictionary=self.data['network_dictionary'],
                                   merchants=self.providers,
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
        node_features = np.array([G.nodes[n]['feature'] for n in G.nodes]).astype(np.float32)
        degrees, eigenvectors = preprocessing.get_nodes_centralities(self.simulator.current_graph)
        if np.max(self.simulator.transaction_amounts) == 0:
            normalized_transaction_amounts = np.zeros_like(self.simulator.transaction_amounts)
        else:
            normalized_transaction_amounts = self.simulator.transaction_amounts / np.sum(self.simulator.transaction_amounts)
        trgs = [self.simulator.map_nodes_to_id[x] for x in self.simulator.trgs]
  
        #set node features
        nodes_list = list(G.nodes())
        for i in range(len(G.nodes())):
            node_features[i][0] = degrees[nodes_list[i]]
            node_features[i][1] = eigenvectors[nodes_list[i]]
            node_features[i][3] = 0
            if i in trgs:
                node_features[i][3] = 1
            node_features[i][4] = normalized_transaction_amounts[i]
        
        # Extract edge index
        edge_index = np.array([(self.simulator.map_nodes_to_id[x],self.simulator.map_nodes_to_id[y]) for (x,y) in G.edges]).T


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

    def get_normalizer_configs(self):
        #return cap_max, base_max, rate_max
        return self.data["fee_base_max"], self.data["fee_rate_max"], self.data["capacity_max"]
    
