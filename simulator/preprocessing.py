import networkx as nx
import pandas as pd
import json
import numpy as np
import random
import math
from operator import itemgetter


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
            # print(f"{vertex} vertex, neigbors: \n {neighbors}")
            new_vertices.update(neighbors)
        
        sampled_vertices = new_vertices.difference(Union_set)
        if len(Union_set) + len(sampled_vertices)>local_size:
            # print(f"Cutting over {local_size}")
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

def make_LN_graph(directed_edges, manual_balance, src, trgs, channel_ids, capacities, initial_balances):
    edges = initiate_balances(directed_edges)
    if manual_balance:
        edges = set_channels_balances(edges, src, trgs, channel_ids, capacities, initial_balances)
    G = nx.from_pandas_edgelist(edges, source="src", target="trg",
                                edge_attr=['channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat', 'balance'],
                               create_using=nx.DiGraph())
    return G

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
    sub_graph = G.subgraph(sub_nodes)
    sub_edges = nx.to_pandas_edgelist(sub_graph)
    sub_edges = sub_edges.rename(columns={'source': 'src', 'target': 'trg'})    
    network_dictionary = create_network_dictionary(sub_edges)

    return network_dictionary, sub_providers, sub_edges, sub_graph


def create_sampled_sub_node(G, src, local_heads_number, providers, local_size, sampling_mode = 'degree'):
    G.add_node(src)
    sub_nodes = set()

    #NOTE: The following were replaced with weighted random sampling
    if sampling_mode == 'degree':
        # random_base_nodes = get_base_nodes_by_degree(G,local_heads_number)
        random_base_nodes =  random_k_nodes_weighted(G, local_heads_number)

    if sampling_mode == 'betweenness':
        random_base_nodes = get_base_nodes_by_betweenness_centrality(G,local_heads_number)
        # random_base_nodes =  random_k_nodes_betweenness_weighted(G, local_heads_number)

    if sampling_mode == 'provider':
        random_base_nodes = get_random_provider(providers, local_heads_number)
        print(f"random providers: {random_base_nodes}")
    
    # NOTE:the following is the previous sampling code
    # for random_base_node in random_base_nodes:
    #         sub_nodes_temp = get_neighbors(G, random_base_node, local_size)
    #         sub_nodes.update(sub_nodes_temp)
    
    #NOTE: the following refers to snowball sampling with choice function being uniform random
    sub_nodes.update(snowball_sampling(G,random_base_nodes,stages=4,k=4, local_size=local_size))
    if len(sub_nodes) < local_size:
        raise GraphTooSmallError()
    # print("subgraph created with size: ",len(sub_nodes)+1)

    
        
    # sub_nodes = set(random.sample(list(sub_nodes), local_size))

    #Check whether the sub nodes we choose for localization is connected or not

    # if is_subgraph_strongly_connected(G, sub_nodes):
    #     print("The subgraph is strongly connected.")
    # else:
    #     print("The subgraph is not strongly connected.")
    #     raise GraphNotConnectedError()
    # print("lengths of components: ", [len(comp) for comp in components(G, sub_nodes)])

    
    sub_nodes.add(src)

    return sub_nodes

def create_list_of_sub_nodes(G, src, local_heads_number, providers, local_size, list_size = 1000):
    max_number_of_iteration = 10000

    list_of_sub_nodes = []
    counter = 0
    while len(list_of_sub_nodes) < list_size or counter == max_number_of_iteration :
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


def init_node_params(edges, providers, verbose=True):
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
    trgs = []#number_of_channels*[None]
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

     
    # #TODO: set back to normal: Has been set back to normal
#  import pickle     
#     with open('output.txt', 'a') as f:
#         # Iterate over the values of 'k'
#         for local_head_numbers in [10,20]:
#             # Iterate over the values of 'stage'
#             for k in [2, 3, 4]:
#                 # Iterate over the values of 'local_head_numbers'
#                 for stage in [1, 2, 3, 4, 5, 10]:
#                     print(" ......... Beginning with k =",k,", stage=",stage," and local_head_numbers=",local_head_numbers," ...........")
#                     sub_nodes_list = []
#                     count = 0
#                     wait = 0
#                     not_connected=0
#                     too_small=0
#                     problem = False
#                     ##
#                     # Initialize an empty list to store the previous results
#                     previous_results = []
#                     # Initialize a counter to count the repetitions
#                     repetitions = 0
#                     ##
#                     while True:
#                         if count == 50:
#                             break
#                         if wait == 5 or not_connected == 5 or too_small == 5:
#                             problem = True
#                             break
                            
#                         #delete  sub_graph output
#                         try:
#                             network_dictionary, nodes, sub_providers, sub_edges, random_base_nodes = create_sub_network(directed_edges, providers, src, trgs,
#                                                                                                  channel_ids, local_size,k,stage,local_head_numbers, manual_balance, initial_balances, capacities)
#                         except GraphNotConnectedError as e:
#                             not_connected+=1
#                             print(e.message, " trying again")
#                             continue
#                         except GraphTooSmallError as e:
#                             too_small+=1
#                             print(e.message, " trying again")
#                             continue
                            
#                         if nodes in sub_nodes_list:
#                             wait = wait+1
#                             print("waiting for new network.....")
#                             continue
#                         #resetting breaking criterias
#                         wait = 0
#                         not_connected=0
#                         too_small=0
                
#                         for node in random_base_nodes:
#                             if node in previous_results:
#                                 repetitions+=1
#                                 print("This Node:", node ,"Has been Seen Before")
#                             else:
#                                 previous_results.append(node)
                        
                        
#                         sub_nodes_list.append(nodes)
#                         count+=1
#                         print("...........Created",count,"ith Sub Network...........")
#                         # active_channels = create_active_channels(network_dictionary, channels)
#                     jacard_scores = []
#                     common_nodes = []
#                     print("The number of created subgraphs is: ", len(sub_nodes_list))
                
#                     for i in range(count):
#                         for j in range(i+1,count):
                
#                             nodes1, nodes2 = sub_nodes_list[i], sub_nodes_list[j]
#                             jacard, common = jaccard_similarity(nodes1, nodes2)
#                             jacard_scores.append(jacard)
#                             common_nodes.append(common)

    
#                     # Write the values to the file
#                     print("........................ WRITING OUTPUT ........................")
#                     if problem == False and len(sub_nodes_list) == 50:
#                         average_jaccard_distances = calculate_average(jacard_scores)
#                         average_common_nodes = calculate_average(common_nodes)
#                         f.write(f'k={k}, stage={stage}, local_head_numbers={local_head_numbers}, average_jaccard_distances={average_jaccard_distances}, average_common_nodes={average_common_nodes}\n')
#                     else:
#                         f.write(f' \n k={k}, stage={stage}, local_head_numbers={local_head_numbers}, Not Connected or too small \n')
                        
        
    
    
        
#         # with open("jacard.txt", "wb") as fp:   
#         #     pickle.dump(jacard_scores, fp)
            
#         # with open("common.txt", "wb") as fp:   
#         #     pickle.dump(common_nodes, fp)
            
    
#         exit()
    balances, capacities = set_channels_balances_and_capacities(src,trgs,network_dictionary)

    
    return active_channels, network_dictionary, node_variables, active_providers, balances, capacities, fee_policy_dict, nodes

def create_fee_policy_dict(directed_edges):
    #get fee_base and fee_rate median for each node
    fee_policy_dict = dict()
    grouped = directed_edges.groupby(["src"])
    temp = grouped.agg({
        "fee_base_msat": "median",
        "fee_rate_milli_msat": "median",
    }).reset_index()[["src","fee_base_msat","fee_rate_milli_msat"]]
    for i in range(len(temp)):
        # fee_policy_dict[temp["src"][i]] = (100, 0.00022) #median fee rates and fee base (sat)
        #NOTE: note that we can use the median of all policies for this
        fee_policy_dict[temp["src"][i]] = (temp["fee_base_msat"][i], temp["fee_rate_milli_msat"][i])

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
    # random.seed(42)
    return random.sample(providers, number_of_heads)

def get_base_nodes_by_degree(G,number_of_heads):
    # random.seed(42)
    top_k_degree_nodes = top_k_nodes(G, number_of_heads)
    return top_k_degree_nodes

def get_base_nodes_by_betweenness_centrality(G,number_of_heads):
    # random.seed(42)
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

def random_k_nodes_weighted(G, k):
    # Compute the degree of each node
    random.seed()
    
    node_degrees = dict(G.degree())
    
    
    # Compute weights based on node degrees
    # maximum_degree = math.log(max(node_degrees.values())+1)
    total_log_degree = sum([math.log(x+1) for x in node_degrees.values()])
    weights = {node: math.log(degree + 1) / total_log_degree for node, degree in node_degrees.items()}

    # total_log_degree = sum([1 for x in node_degrees.values()])
    # weights = {node: 1 / total_log_degree for node, degree in node_degrees.items()}

    # Sample k nodes with weighted randomness

    sampled_nodes = random.choices(list(weights.keys()), weights=list(weights.values()), k=k)

    return sampled_nodes

def random_k_nodes_betweenness_weighted(G, k):
    # Compute the betweenness centrality of each node
    node_betweenness = nx.betweenness_centrality(G)

    # Compute weights based on betweenness centrality
    total_betweenness = sum(node_betweenness.values())
    weights = {node: centrality / total_betweenness for node, centrality in node_betweenness.items()}

    # Sample k nodes with weighted randomness
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
