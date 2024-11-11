import networkx as nx
import pandas as pd
import json
import numpy as np
import random
import math
from operator import itemgetter
from collections import deque

def aggregate_edges(directed_edges):
    """aggregating multiedges"""
    grouped = directed_edges.groupby(["src", "trg"])
    directed_aggr_edges = grouped.agg({
        "capacity": "sum",
        "fee_base_msat": "mean",
        "fee_rate_milli_msat": "mean",
        "last_update": "max",
        "channel_id": "first",
        "disabled": "first",
        "min_htlc": "mean",
    }).reset_index()
    return directed_aggr_edges

def fireforest_sample(G, sample_size, providers, local_heads_number, p=0.3):
    """
    Performs a fire forest sampling algorithm to select a sample of nodes from the given graph `G`.
    
    Args:
        G (networkx.Graph): The input graph to sample from.
        sample_size (int): The desired size of the sample.
        providers (list): A list of provider nodes to start the sampling from.
        local_heads_number (int): The number of local heads to select from the providers.
        p (float, optional): The probability of burning a neighbor node during the sampling process. Defaults to 0.7.
    
    Returns:
        list: A list of sampled nodes.
    """
    # random.seed(44)
        
    sampled_nodes = set()
    while len(sampled_nodes) < sample_size:

        burning_nodes = get_random_provider(providers, local_heads_number)
    
        while burning_nodes and len(sampled_nodes) < sample_size:
            current_node = burning_nodes.pop(0)
            if current_node not in sampled_nodes:
                sampled_nodes.add(current_node)
                # Burn neighbors with probability p
                neighbors = list(G.neighbors(current_node))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in sampled_nodes and random.random() < p:
                        burning_nodes.append(neighbor)

        #check connectivity and size        
        if len(sampled_nodes) < sample_size and not is_subgraph_connected(G, sampled_nodes):
            sampled_nodes = set()
            # random.seed(17)

    return sorted(list(sampled_nodes))

def initiate_balances(directed_edges, approach='half'):
    '''
    approach = 'random'
    approach = 'half'


    NOTE : This Function is written assuming that two side of channels are next to each other in directed_edges
    '''
    G = directed_edges[['src', 'trg', 'channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat']]
    G = G.assign(balance=None)
    r = 0.5
    for index, row in G.iterrows():
        balance = 0
        cap = row['capacity']
        if index % 2 == 0:
            if approach == 'random':
                r = np.random.random()
            balance = r * cap
        else:
            balance = (1 - r) * cap
        G.at[index, "balance"] = balance

    return G

def set_channels_balances(edges, src, trgs, channel_ids, capacities, initial_balances):
    if (len(trgs) == len(capacities)) & (len(trgs) == len(initial_balances)):
        for i in range(len(trgs)):
            trg = trgs[i]
            capacity = capacities[i]
            initial_balance = initial_balances[i]
            index = edges.index[(edges['src'] == src) & (edges['trg'] == trg)]
            reverse_index = edges.index[(edges['src'] == trg) & (edges['trg'] == src)]

            edges.at[index[0], 'capacity'] = capacity
            edges.at[index[0], 'balance'] = initial_balance
            edges.at[reverse_index[0], 'capacity'] = capacity
            edges.at[reverse_index[0], 'balance'] = capacity - initial_balance

        return edges
    else:
        print("Error : Invalid Input Length")

def create_network_dictionary(G):
    """
    Creates a dictionary that maps each channel (represented as a tuple of source and target nodes) to a list containing the channel's balance, fee rate, fee base, and capacity.
    
    Args:
        G (pandas.DataFrame): A DataFrame containing the network's edges, with columns for 'src', 'trg', 'balance', 'fee_rate_milli_msat', 'fee_base_msat', and 'capacity'.
    
    Returns:
        dict: A dictionary mapping each channel to a list of its properties.
    """
    keys = list(zip(G["src"], G["trg"]))
    vals = [list(item) for item in zip(G["balance"], G["fee_rate_milli_msat"], G['fee_base_msat'], G["capacity"])]
    network_dictionary = dict(zip(keys, vals))
    return network_dictionary

def make_LN_graph(directed_edges, providers):
    edges = initiate_balances(directed_edges)
    
    G = nx.from_pandas_edgelist(edges, source="src", target="trg",
                                edge_attr=['channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat', 'balance'],
                               create_using=nx.DiGraph())
    
    #NOTE: the node features vector is as follows: [degree_centrality, is_provider, is_connected_to_us,
    # total budget, transaction amount]
    # degrees, closeness, eigenvectors = set_node_attributes(G)
    providers_nodes = list(set(providers))
    
    
    for node in G.nodes():
        G.nodes[node]["feature"] = np.array([0, node in providers_nodes, 0, 0])
    return G

def get_nodes_degree_centrality(G):
    degrees = nx.degree_centrality(G)
    return degrees

def get_sub_graph_properties(G, sub_nodes, providers):
    """
    Extracts the properties of a subgraph from the given graph `G` and the set of subgraph nodes `sub_nodes`.
    
    Args:
        G (networkx.Graph): The full graph.
        sub_nodes (set): The set of nodes in the subgraph.
        providers (list): The list of provider nodes.
    
    Returns:
        tuple:
            - network_dictionary (dict): A dictionary mapping node pairs to a list of edge attributes.
            - sub_providers (list): The list of provider nodes in the subgraph.
            - sub_edges (pandas.DataFrame): The DataFrame of edges in the subgraph.
            - sub_graph (networkx.Graph): The subgraph.
    """
        
    
    sub_providers = list(set(sub_nodes) & set(providers))
    sub_graph = G.subgraph(sub_nodes).copy()
    degrees = get_nodes_degree_centrality(G)

    #set centrality of nodes
    for node in G.nodes():
        G.nodes[node]["feature"][0] = degrees[node]
        
    sub_edges = nx.to_pandas_edgelist(sub_graph)
    sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})  
    network_dictionary = create_network_dictionary(sub_edges)

    return network_dictionary, sub_providers, sub_edges, sub_graph

def components(G, nodes):
    H = G.subgraph(nodes)
    return nx.strongly_connected_components(H)

def init_node_params(edges, providers, verbose=False):

    """Initialize source and target distribution of each node in order to draw transaction at random later."""
    
    G = nx.from_pandas_edgelist(edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.DiGraph())
    active_providers = list(set(providers).intersection(set(G.nodes())))
    if len(providers) == 0:
        active_ratio = 0
    else:
        active_ratio = len(active_providers) / len(providers)
    if verbose:
        print("Total number of possible providers: %i" % len(providers))
        print("Ratio of active providers: %.2f" % active_ratio)
    degrees = pd.DataFrame(list(G.degree()), columns=["pub_key", "degree"])
    total_capacity = pd.DataFrame(list(nx.degree(G, weight="capacity")), columns=["pub_key", "total_capacity"])
    node_variables = degrees.merge(total_capacity, on="pub_key")
    return node_variables, active_providers, active_ratio

def get_providers(providers_path):
    """
    Retrieves a list of provider public keys from a JSON file.
    
    Args:
        providers_path (str): The file path to the JSON file containing the provider public keys.
    
    Returns:
        list: A list of provider public keys.
    """
    with open(providers_path) as f:
        tmp_json = json.load(f)
    providers = []
    for i in range(len(tmp_json)):
        providers.append(tmp_json[i].get('pub_key'))
    return providers

def get_directed_edges(directed_edges_path):
    """
    Retrieves a DataFrame of directed edges from a JSON file.
    
    Args:
        directed_edges_path (str): The file path to the JSON file containing the directed edges.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the directed edges, with columns 'src', 'trg', and 'channel_id'.
    """
    directed_edges = pd.read_json(directed_edges_path)
    directed_edges = aggregate_edges(directed_edges)
    return directed_edges

def select_node(directed_edges, src_index):
    src = directed_edges.iloc[src_index]['src']
    trgs = directed_edges.loc[(directed_edges['src'] == src)]['trg']
    channel_ids = directed_edges.loc[(directed_edges['src'] == src)]['channel_id']
    number_of_channels = len(trgs)
    return src, list(trgs), list(channel_ids), number_of_channels 
#NOTE: the followings are to check the similarity of graphs
def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3), len(i)

def graph_edit_distance_similarity(graph1, graph2):
    # Compute the graph edit distance
    ged = nx.graph_edit_distance(graph1, graph2)
    
    # Normalize the graph edit distance to obtain a similarity score
    max_possible_ged = max(len(graph1.edges()), len(graph2.edges()))
    similarity = 1 - (ged / max_possible_ged)
    
    return ged,similarity

def create_fee_policy_dict(directed_edges, src):
    """
    Creates a dictionary that maps each source node to a tuple of the median fee base and fee rate for that node's outgoing channels.
    
    Args:
        directed_edges (pandas.DataFrame): A DataFrame containing the directed edges, with columns 'src', 'trg', 'fee_base_msat', and 'fee_rate_milli_msat'.
        src (str): The source node for which to create the fee policy dictionary.
    
    Returns:
        dict: A dictionary that maps each source node to a tuple of the median fee base and fee rate for that node's outgoing channels.
    """
    fee_policy_dict = dict()
    
    median_base = directed_edges["fee_base_msat"].median()
    median_rate = directed_edges["fee_rate_milli_msat"].median()
    grouped = directed_edges.groupby(["src"])
    temp = grouped.agg({
        "fee_base_msat": "median",
        "fee_rate_milli_msat": "median",
    }).reset_index()
    for i in range(len(temp)):
        fee_policy_dict[temp["src"][i]] = (temp["fee_base_msat"][i], temp["fee_rate_milli_msat"][i])
    fee_policy_dict[src] = (median_base, median_rate)
    return fee_policy_dict

def set_channels_balances_and_capacities(src,trgs,network_dictionary):
    balances = []
    capacities = []
    for trg in trgs:
        b = network_dictionary[(src, trg)][0]
        c = network_dictionary[(src, trg)][3]
        balances.append(b)
        capacities.append(c)
    return balances, capacities

def generate_transaction_types(number_of_transaction_types, counts, amounts, epsilons):
    transaction_types = []
    for i in range(number_of_transaction_types):
        transaction_types.append((counts[i], amounts[i], epsilons[i]))
    return transaction_types

def get_random_provider(providers, number_of_heads):
    # random.seed(44)
    random.seed()
    return random.sample(providers, number_of_heads)

def get_base_nodes_by_degree(G,number_of_heads):
    top_k_degree_nodes = top_k_nodes(G, number_of_heads)
    return top_k_degree_nodes

def get_base_nodes_by_betweenness_centrality(G,number_of_heads):
    top_k_betweenness_centrality_nodes = top_k_nodes_betweenness(G, number_of_heads)
    return top_k_betweenness_centrality_nodes

def top_k_nodes(G, k):
    # Compute the degree of each node
    node_degrees = G.degree()
    
    # Sort nodes by degree
    sorted_nodes = sorted(node_degrees, key=itemgetter(1), reverse=True)
    
    # Get the top k nodes
    top_k = sorted_nodes[:k]
    
    # Return only the nodes, not their degrees
    return [node for node, degree in top_k]

def random_k_nodes_log_weighted(G, k):
    # Compute the degree of each node
    random.seed()
    
    node_degrees = dict(G.degree())
    
    total_log_degree = sum([math.log(x+1) for x in node_degrees.values()])
    weights = {node: math.log(degree + 1) / total_log_degree for node, degree in node_degrees.items()}

    sampled_nodes = random.choices(list(weights.keys()), weights=list(weights.values()), k=k)

    return sampled_nodes

def random_k_nodes_betweenness_weighted(G, k):
    random.seed()
    
    node_betweenness = nx.betweenness_centrality(G)

    total_betweenness = sum(node_betweenness.values())
    weights = {node: centrality / total_betweenness for node, centrality in node_betweenness.items()}

    sampled_nodes = random.choices(list(weights.keys()), weights=list(weights.values()), k=k)

    return sampled_nodes

def top_k_nodes_betweenness(G, k):
    # Compute the betweenness centrality of each node
    node_betweenness = nx.betweenness_centrality(G)
    
    # Sort nodes by betweenness centrality
    sorted_nodes = sorted(node_betweenness.items(), key=itemgetter(1), reverse=True)
    
    # Get the top k nodes
    top_k = sorted_nodes[:k]
    
    # Return only the nodes, not their betweenness centrality
    return [node for node, centrality in top_k]


def is_subgraph_connected(G, nodes):
    H = G.subgraph(nodes)
    return nx.is_connected(H)



class GraphNotConnectedError(Exception):
    """Exception raised when the graph is not connected."""
    
    def __init__(self, message="Graph is not connected"):
        self.message = message
        super().__init__(self.message)

class GraphTooSmallError(Exception):
    """Exception raised when the graph size is less than expected."""
    
    def __init__(self, message="Finall graph is too small."):
        self.message = message
        super().__init__(self.message)
