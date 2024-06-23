import networkx as nx
import pandas as pd
import json
import numpy as np
import random
import math
from operator import itemgetter
from littleballoffur import ForestFireSampler
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

    
def get_neighbors(G, src, local_size):
    """localising the network around the node"""

    neighbors = [src]

    for i in range(10):
        outer_list = []
        for neighbor in neighbors:
            inner_list = list(G.neighbors(neighbor))

            for v in inner_list:

               if len(neighbors) > local_size:
                  print('size of sub network: ', len(neighbors))
                  return set(neighbors)
               if v not in neighbor:
                  neighbors.append(v)

def bfs_k_levels(G, src, k):
    """Localize the network around the node up to k levels using BFS"""

    # Initialize a set to store the nodes visited
    neighbors = set([src])

    # Initialize a queue for BFS
    queue = [(src, 0)]

    while queue:
        node, level = queue.pop(0)

        if level == k:
            break

        for neighbor in G.neighbors(node):
            if neighbor not in neighbors:
                neighbors.add(neighbor)
                queue.append((neighbor, level + 1))

    print('Number of nodes in k-level BFS: ', len(neighbors))
    return neighbors


def snowball_sampling(G, initial_vertices, stages, k, local_size):
    """
    Perform snowball sampling on a graph G.

    Parameters:
        G (networkx.Graph): The graph to sample from.
        initial_vertices (list): Initial set of vertices V(0).
        stages (int): Number of stages for the sampling process.
        k (int): Number of neighboring nodes to query at each stage.

    Returns:
        set: Set of sampled vertices.
    """
    random.seed()

    # print(f"initial_vertices: {initial_vertices}")
    Union_set = set(initial_vertices)
    sampled_vertices = set(initial_vertices)
    
    for i in range(1, stages + 1):
        new_vertices = set()

        if len(Union_set)>= local_size:
            break
        for vertex in sampled_vertices:
            neighbors = get_snowball_neighbors(G, vertex, k)
            new_vertices.update(neighbors)
        
        sampled_vertices = new_vertices.difference(Union_set)
        if len(Union_set) + len(sampled_vertices)>local_size:
            Union_set.update(set(random.sample(list(sampled_vertices), local_size - len(Union_set))))
            break
        Union_set.update(new_vertices)

    return Union_set

def get_snowball_neighbors(G, vertex, k):
    
    """localising the network around the node"""
    random.seed()
    
    neighbors = list(G.neighbors(vertex))
    sampled_neighbors = random.sample(neighbors, min(k, len(neighbors)))
    
    return set(sampled_neighbors)


def get_fire_forest_sample(G, reverse_mapping, sample_size, burning_prob=0.7):
    # Step 1: Convert the directed graph to an undirected graph
    
    
    # # Adding edges and attributes to the undirected graph, avoiding duplicates
    # for u, v, data in G.edges(data=True):
    #     if not undirected_G.has_edge(u, v):
    #         undirected_G.add_edge(u, v, **data)
    
    # # Adding node attributes
    # for node, data in G.nodes(data=True):
    #     undirected_G.nodes[node].update(data)
    

    

    
    # Step 2: Apply the ForestFireSampler to the undirected graph
    forestFireSampler = ForestFireSampler(number_of_nodes=sample_size, p=burning_prob,
                                           max_visited_nodes_backlog=100, restart_hop_size=10)
    random.seed()
    # Wrapper around the sample method
    def wrapped_sample(graph):
        def modified_start_a_fire(graph):
            remaining_nodes = list(forestFireSampler._set_of_nodes.difference(forestFireSampler._sampled_nodes))
            seed_node = random.choice(remaining_nodes)
            forestFireSampler._sampled_nodes.add(seed_node)
            node_queue = deque([seed_node])
            while len(forestFireSampler._sampled_nodes) < forestFireSampler.number_of_nodes:
                if len(node_queue) == 0:
                    node_queue = deque(
                        [
                            forestFireSampler._visited_nodes.popleft()
                            for k in range(
                                min(forestFireSampler.restart_hop_size, len(forestFireSampler._visited_nodes))
                            )
                        ]
                    )
                    if len(node_queue) == 0:
                        print(
                            "Warning: could not collect the required number of nodes. The fire could not find enough nodes to burn."
                        )
                        break
                top_node = node_queue.popleft()
                forestFireSampler._sampled_nodes.add(top_node)
                neighbors = set(forestFireSampler.backend.get_neighbors(graph, top_node))
                unvisited_neighbors = neighbors.difference(forestFireSampler._sampled_nodes)
                score = np.random.geometric(forestFireSampler.p)
                count = min(len(unvisited_neighbors), score)
                burned_neighbors = random.sample(list(unvisited_neighbors), count)  # Convert to list here
                forestFireSampler._visited_nodes.extendleft(
                    unvisited_neighbors.difference(set(burned_neighbors))
                )
                for neighbor in burned_neighbors:
                    if len(forestFireSampler._sampled_nodes) >= forestFireSampler.number_of_nodes:
                        break
                    node_queue.extend([neighbor])
        
        # Override the _start_a_fire method
        forestFireSampler._start_a_fire = modified_start_a_fire
        while True:
            sampled_nodes = forestFireSampler.sample(graph)
            if len(sampled_nodes) >= sample_size:
                return sampled_nodes
            else:
                print("Resampling due to insufficient sample size.")
                random.seed()
    
    sampled_numeric_undirected_G = wrapped_sample(G)
    # Map back to original node labels
    
    sampled_undirected_G = nx.relabel_nodes(sampled_numeric_undirected_G, reverse_mapping)
    sub_nodes = list(sampled_undirected_G.nodes())
    return sub_nodes
    

    

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
    keys = list(zip(G["src"], G["trg"]))
    vals = [list(item) for item in zip([None] * len(G), G["fee_rate_milli_msat"], G['fee_base_msat'], G["capacity"])]

    network_dictionary = dict(zip(keys, vals))
    for index, row in G.iterrows():
        src = row['src']
        trg = row['trg']
        network_dictionary[(src, trg)][0] = row['balance']

    return network_dictionary


def create_active_channels(network_dictionary, channels):
    # channels = [(src1,trg1),(src2,trg2),...]
    active_channels = dict()
    for (src, trg) in channels:
        active_channels[(src, trg)] = network_dictionary[(src, trg)]
        active_channels[(trg, src)] = network_dictionary[(trg, src)]
    return active_channels

def make_LN_graph(directed_edges, providers, manual_balance, src, trgs, channel_ids, capacities, initial_balances,):
    edges = initiate_balances(directed_edges)
    if manual_balance:
        edges = set_channels_balances(edges, src, trgs, channel_ids, capacities, initial_balances)
    G = nx.from_pandas_edgelist(edges, source="src", target="trg",
                                edge_attr=['channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat', 'balance'],
                               create_using=nx.DiGraph())
    
    #NOTE: the node features vector is as follows: [degree_centrality, eigenvectors_centrality, is_provider, is_connected_to_us, normalized_transaction_amount]
    # degrees, closeness, eigenvectors = set_node_attributes(G)
    providers_nodes = list(set(providers))
    for node in G.nodes():
        G.nodes[node]["feature"] = np.array([0, 0, node in providers_nodes, 0, 0])
    return G

def get_nodes_centralities(G):
    degrees = nx.degree_centrality(G)
    eigenvectors = get_eigenvector_centrality(G, degrees)
    return degrees, eigenvectors

def get_eigenvector_centrality(G, degree_centrality):
    try:
        # Try calculating eigenvector centrality with default parameters
        eigenvectors = nx.eigenvector_centrality(G)
        return eigenvectors
    except nx.PowerIterationFailedConvergence as e:
        print(f"Eigenvector centrality failed to converge: {e}. Trying with increased iterations and adjusted tolerance.")
        try:
            # Try calculating eigenvector centrality with increased iterations and adjusted tolerance
            eigenvectors = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
            return eigenvectors
        except nx.PowerIterationFailedConvergence as e:
            print(f"Eigenvector centrality still failed to converge: {e}. Using degree centrality as a fallback.")
            return degree_centrality
    

def create_sub_network(directed_edges, providers, src, trgs, channel_ids, local_size, local_heads_number, manual_balance=False, initial_balances = [], capacities=[]):
    """creating network_dictionary, edges and providers for the local subgraph."""
    print("............creating network_dictionary.................")

    G = make_LN_graph(directed_edges, manual_balance, src, trgs, channel_ids, capacities, initial_balances)

    if len(trgs)==0:
        print("No trgs found")
        #NOTE: in CHANNEL OPENNING case, instead of src, a provider is given for generating the local subgraph
        sub_nodes = create_sampled_sub_node(G,src,local_heads_number,providers,local_size,sampling_mode = 'degree')
        
    else:
        sub_nodes = get_neighbors(G, src, local_size)
        
    network_dictionary, sub_providers, sub_edges, _ = get_sub_graph_properties(G,sub_nodes,providers)

    # network_dictionary = {(src,trg):[balance,alpha,beta,capacity]}

    return network_dictionary, sub_nodes, sub_providers, sub_edges

def get_sub_graph_properties(G,sub_nodes,providers):

    sub_providers = list(set(sub_nodes) & set(providers))
    sub_graph = G.subgraph(sub_nodes).copy()
    degrees, eigenvectors = get_nodes_centralities(G)
    #set centrality of nodes
    for node in G.nodes():
        G.nodes[node]["feature"][:2] = degrees[node], eigenvectors[node]
        
    sub_edges = nx.to_pandas_edgelist(sub_graph)
    sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})    
    network_dictionary = create_network_dictionary(sub_edges)

    return network_dictionary, sub_providers, sub_edges, sub_graph


def create_sampled_sub_node(G, src, local_heads_number, providers, local_size, sampling_mode = 'degree'):
    G.add_node(src)
    sub_nodes = set()

    #NOTE: The following were replaced with weighted random sampling
    if sampling_mode == 'degree':
        random_base_nodes =  random_k_nodes_log_weighted(G, local_heads_number)

    if sampling_mode == 'betweenness':
        random_base_nodes =  random_k_nodes_betweenness_weighted(G, local_heads_number)

    if sampling_mode == 'provider':
        random_base_nodes = get_random_provider(providers, local_heads_number)
    
    
    sub_nodes.update(snowball_sampling(G,random_base_nodes,stages=4,k=4, local_size=local_size))
    
    if len(sub_nodes) < local_size:
        raise GraphTooSmallError()

    if not is_subgraph_strongly_connected(G, sub_nodes):
        raise GraphNotConnectedError()
    
    sub_nodes.add(src)

    return sub_nodes

def create_list_of_sub_nodes(G, src, local_heads_number, providers, local_size, list_size = 5000):
    max_number_of_iteration = 10000

    list_of_sub_nodes = []
    counter = 0
    while len(list_of_sub_nodes) < list_size and counter <= max_number_of_iteration :
        try:
            sub_node = create_sampled_sub_node(G, src, local_heads_number, providers, local_size, sampling_mode = 'degree')
            if sub_node not in list_of_sub_nodes:
                list_of_sub_nodes.append(sub_node)
                print("Added:-->",len(list_of_sub_nodes))
            else:
                print("This Graph has been created before")
            counter+=1
        except GraphNotConnectedError as e:
            print(e.message, " trying again")
            counter+=1
            continue
        except GraphTooSmallError as e:
            print(e.message, " trying again")
            counter+=1
            continue

    return list_of_sub_nodes


def components(G, nodes):
    H = G.subgraph(nodes)
    return nx.strongly_connected_components(H)


def init_node_params(edges, providers, verbose=False):
    """Initialize source and target distribution of each node in order to draw transaction at random later."""
    G = nx.from_pandas_edgelist(edges, source="src", target="trg", edge_attr=["capacity"], create_using=nx.DiGraph())
    active_providers = list(set(providers).intersection(set(G.nodes())))
    active_ratio = len(active_providers) / len(providers)
    if verbose:
        print("Total number of possible providers: %i" % len(providers))
        print("Ratio of active providers: %.2f" % active_ratio)
    degrees = pd.DataFrame(list(G.degree()), columns=["pub_key", "degree"])
    total_capacity = pd.DataFrame(list(nx.degree(G, weight="capacity")), columns=["pub_key", "total_capacity"])
    node_variables = degrees.merge(total_capacity, on="pub_key")
    return node_variables, active_providers, active_ratio


def get_providers(providers_path):
    # The path should direct this to a json file containing providers
    with open(providers_path) as f:
        tmp_json = json.load(f)
    providers = []
    for i in range(len(tmp_json)):
        providers.append(tmp_json[i].get('pub_key'))
    return providers


def get_directed_edges(directed_edges_path):
    directed_edges = pd.read_json(directed_edges_path)
    directed_edges = aggregate_edges(directed_edges)
    return directed_edges


def select_node(directed_edges, src_index):
    src = directed_edges.iloc[src_index]['src']
    trgs = directed_edges.loc[(directed_edges['src'] == src)]['trg']
    channel_ids = directed_edges.loc[(directed_edges['src'] == src)]['channel_id']
    number_of_channels = len(trgs)
    return src, list(trgs), list(channel_ids), number_of_channels

#NOTE: creates the node for channel openning mode
def create_node(directed_edges, src, number_of_channels):
    trgs = []
    max_id = max(directed_edges['channel_id'])
    channel_ids = [(max_id + i + 1) for i in range (number_of_channels*2)]
    return src, list(trgs), list(channel_ids)
    

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
    
def get_init_parameters(providers, directed_edges, src, trgs, channel_ids, channels, local_size, manual_balance, initial_balances,capacities,mode, local_heads_number):
    fee_policy_dict = create_fee_policy_dict(directed_edges)     
    
    network_dictionary, nodes, sub_providers, sub_edges = create_sub_network(directed_edges, providers, src, trgs,
                                                                             channel_ids, local_size, local_heads_number, manual_balance, initial_balances, capacities)
    active_channels = create_active_channels(network_dictionary, channels)

    try:
        node_variables, active_providers, active_ratio = init_node_params(sub_edges, sub_providers, verbose=True)
    except:
        print('zero providers!')

    balances, capacities = set_channels_balances_and_capacities(src,trgs,network_dictionary)

    
    return active_channels, network_dictionary, node_variables, active_providers, balances, capacities, fee_policy_dict, nodes

def create_fee_policy_dict(directed_edges, src):
    #get fee_base and fee_rate median for each node
    fee_policy_dict = dict()
    
    median_base = directed_edges["fee_base_msat"].median()
    median_rate = directed_edges["fee_rate_milli_msat"].median()
    
    grouped = directed_edges.groupby(["src"])
    temp = grouped.agg({
        "fee_base_msat": "median",
        "fee_rate_milli_msat": "median",
    }).reset_index()[["src","fee_base_msat","fee_rate_milli_msat"]]
    
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

def is_subgraph_weakly_connected(G, nodes):
    """
    Check if the subgraph induced by 'nodes' in directed graph 'G' is weakly connected.

    Parameters:
    G (networkx.DiGraph): The main directed graph.
    nodes (list): The nodes of the subgraph.

    Returns:
    bool: True if the subgraph is weakly connected, False otherwise.
    """
    H = G.subgraph(nodes)
    return nx.is_weakly_connected(H)

def is_subgraph_strongly_connected(G, nodes):
    """
    Check if the subgraph induced by 'nodes' in directed graph 'G' is strongly connected.

    Parameters:
    G (networkx.DiGraph): The main directed graph.
    nodes (list): The nodes of the subgraph.

    Returns:
    bool: True if the subgraph is strongly connected, False otherwise.
    """
    H = G.subgraph(nodes)
    return nx.is_strongly_connected(H)



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
