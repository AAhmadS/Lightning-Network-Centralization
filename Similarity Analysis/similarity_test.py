

import random
import pandas as pd
import networkx as nx
import numpy as np
import json
from scipy.stats import ks_2samp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def get_random_provider(providers, number_of_heads):
    random.seed()
    return random.sample(providers, number_of_heads)

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

def is_subgraph_connected(G, nodes):
    H = G.subgraph(nodes)
    return nx.is_connected(H)

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

def make_LN_graph(directed_edges, providers):
    edges = initiate_balances(directed_edges)
    
    G = nx.from_pandas_edgelist(edges, source="src", target="trg",
                                edge_attr=['channel_id', 'capacity', 'fee_base_msat', 'fee_rate_milli_msat', 'balance'],
                               create_using=nx.DiGraph())
    
    # the node features vector is as follows: [degree_centrality, is_provider, is_connected_to_us,
    # total budget, transaction amount]
    # degrees, closeness, eigenvectors = set_node_attributes(G)
    providers_nodes = list(set(providers))
    
    
    for node in G.nodes():
        G.nodes[node]["feature"] = np.array([0, node in providers_nodes, 0, 0])
    return G

def set_undirected_attributed_LN_graph(G):
        """    
        Sets the undirected attributed Lightning Network (LN) graph for the environment.
        
        Returns:
            networkx.Graph: The undirected attributed LN graph.
        """
        undirected_G = nx.Graph(G)
        return undirected_G


directed_edges = get_directed_edges("data.json")
providers = get_providers("merchants.json")
G = make_LN_graph(directed_edges, providers)
undirected_G = set_undirected_attributed_LN_graph(G)


# Analysis Functions for S1-S9 Graph Properties

def compute_degree_distribution(G):
    """S1/S2: Degree distribution (for undirected graphs, in-degree = out-degree = degree)"""
    if G.is_directed():
        # For directed graphs, compute both in and out degrees
        in_degrees = [G.in_degree(node) for node in G.nodes()]
        out_degrees = [G.out_degree(node) for node in G.nodes()]
        return {
            'in_degree': np.array(list(Counter(in_degrees).values())),
            'out_degree': np.array(list(Counter(out_degrees).values()))
        }
    else:
        # For undirected graphs, degree = in_degree = out_degree
        degrees = [G.degree(node) for node in G.nodes()]
        degree_dist = np.array(list(Counter(degrees).values()))
        return {
            'in_degree': degree_dist,  # Same as degree for undirected
            'out_degree': degree_dist  # Same as degree for undirected
        }

def compute_connected_components_distribution(G):
    """S3/S4: Connected components distribution"""
    if G.is_directed():
        # For directed graphs, compute both weakly and strongly connected components
        wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
        scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
        return {
            'wcc': np.array(list(Counter(wcc_sizes).values())),
            'scc': np.array(list(Counter(scc_sizes).values()))
        }
    else:
        # For undirected graphs, there's only one type of connected component
        cc_sizes = [len(c) for c in nx.connected_components(G)]
        cc_dist = np.array(list(Counter(cc_sizes).values()))
        return {
            'wcc': cc_dist,  # Same as connected components for undirected
            'scc': cc_dist   # Same as connected components for undirected
        }

def compute_hop_plot(G, max_hops=10, is_original_graph=False):
    """S5: Hop-plot - number of reachable pairs at distance h or less"""
    import time
    
    hop_counts = []
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Only sample nodes for subgraphs, not for the original full graph
    if not is_original_graph and n_nodes > 1000:
        nodes = random.sample(nodes, min(1000, n_nodes))
        print(f"  Sampling {len(nodes)} nodes for hop-plot computation (subgraph analysis)")
    
    print(f"  🗺️  Starting hop-plot computation for {len(nodes)} nodes, {max_hops} hops")
    
    for h in range(1, max_hops + 1):
        hop_start = time.time()
        reachable_pairs = 0
        
        print(f"    Computing hop {h}/{max_hops}...")
        
        for i, node in enumerate(nodes):
            # Progress logging every 1000 nodes or every 10% of progress
            if i > 0 and (i % 1000 == 0 or i % max(1, len(nodes) // 10) == 0):
                elapsed = time.time() - hop_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(nodes) - i) / rate if rate > 0 else 0
                progress_pct = (i / len(nodes)) * 100
                print(f"      Progress: {i}/{len(nodes)} nodes ({progress_pct:.1f}%) | "
                      f"Rate: {rate:.1f} nodes/sec | ETA: {eta:.0f}s")
            
            try:
                paths = nx.single_source_shortest_path_length(G, node, cutoff=h)
                reachable_pairs += len(paths) - 1  # exclude the node itself
            except:
                continue
        
        hop_time = time.time() - hop_start
        print(f"    ✓ Hop {h} completed in {hop_time:.1f}s | Reachable pairs: {reachable_pairs:,}")
        hop_counts.append(reachable_pairs)
    
    return np.array(hop_counts)

def compute_hop_plot_largest_wcc(G, max_hops=10, is_original_graph=False):
    """S6: Hop-plot on largest weakly connected component"""
    if G.is_directed():
        wcc = max(nx.weakly_connected_components(G), key=len)
    else:
        wcc = max(nx.connected_components(G), key=len)
    G_wcc = G.subgraph(wcc)
    return compute_hop_plot(G_wcc, max_hops, is_original_graph=is_original_graph)

def compute_singular_vector_distribution(G, is_original_graph=False):
    """S7: Distribution of first left singular vector vs rank"""
    try:
        A = nx.adjacency_matrix(G)
        
        # Only sample for subgraphs, not for the original full graph
        if not is_original_graph and A.shape[0] > 1000:
            nodes = random.sample(list(G.nodes()), min(1000, A.shape[0]))
            G_sample = G.subgraph(nodes)
            A = nx.adjacency_matrix(G_sample)
            print(f"  Sampling {len(nodes)} nodes for SVD computation (subgraph analysis)")
        
        # For original graph, use more singular vectors; for samples, use fewer
        k = min(100 if is_original_graph else 50, A.shape[0] - 1)
        if k > 0:
            # Add regularization to handle sparse matrices
            if A.nnz == 0:
                return np.zeros(A.shape[0])
            A_reg = A + 1e-12 * np.eye(A.shape[0])
            u, s, vt = svds(A_reg.astype(float), k=k)
            return np.abs(u[:, 0])  # First left singular vector
        else:
            return np.array([0])
    except Exception as e:
        print(f"  SVD computation failed: {e}")
        return np.array([0])

def compute_singular_values_distribution(G, is_original_graph=False):
    """S8: Distribution of singular values vs rank"""
    try:
        A = nx.adjacency_matrix(G)
        
        # Only sample for subgraphs, not for the original full graph
        if not is_original_graph and A.shape[0] > 1000:
            nodes = random.sample(list(G.nodes()), min(1000, A.shape[0]))
            G_sample = G.subgraph(nodes)
            A = nx.adjacency_matrix(G_sample)
            print(f"  Sampling {len(nodes)} nodes for singular values computation (subgraph analysis)")
        
        # For original graph, use more singular vectors; for samples, use fewer
        k = min(100 if is_original_graph else 50, A.shape[0] - 1)
        if k > 0:
            # Add regularization to handle sparse matrices
            if A.nnz == 0:
                return np.array([0])
            A_reg = A + 1e-12 * np.eye(A.shape[0])
            u, s, vt = svds(A_reg.astype(float), k=k)
            return s
        else:
            return np.array([0])
    except Exception as e:
        print(f"  Singular values computation failed: {e}")
        return np.array([0])

def compute_clustering_coefficient_distribution(G):
    """S9: Clustering coefficient distribution by degree"""
    # Convert to undirected for clustering coefficient
    G_undirected = G.to_undirected() if G.is_directed() else G
    clustering = nx.clustering(G_undirected)
    degrees = dict(G_undirected.degree())
    
    degree_clustering = defaultdict(list)
    for node in G_undirected.nodes():
        degree = degrees[node]
        degree_clustering[degree].append(clustering[node])
    
    # Average clustering coefficient for each degree
    avg_clustering_by_degree = []
    for degree in sorted(degree_clustering.keys()):
        avg_clustering_by_degree.append(np.mean(degree_clustering[degree]))
    
    return np.array(avg_clustering_by_degree)

def compute_all_graph_properties(G, is_original_graph=False):
    """Compute all S1-S9 properties for a graph"""
    properties = {}
    
    graph_type = "original" if is_original_graph else "sample"
    print(f"   Computing properties for {graph_type} graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    
    import time
    start_time = time.time()
    
    try:
        print("     Computing S1/S2: Degree distributions...")
        degree_dists = compute_degree_distribution(G)
        properties['in_degree'] = degree_dists['in_degree']
        properties['out_degree'] = degree_dists['out_degree']
        print(f"       ✓ S1/S2 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"     S1/S2 (Degree distributions) failed: {e}")
        properties['in_degree'] = np.array([0])
        properties['out_degree'] = np.array([0])
    
    try:
        print("    🔗 Computing S3/S4: Connected components...")
        cc_dists = compute_connected_components_distribution(G)
        properties['wcc'] = cc_dists['wcc']
        properties['scc'] = cc_dists['scc']
        print(f"       ✓ S3/S4 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"     S3/S4 (Connected components) failed: {e}")
        properties['wcc'] = np.array([0])
        properties['scc'] = np.array([0])
    
    try:
        print("      Computing S5: Hop-plot...")
        s5_start = time.time()
        properties['hop_plot'] = compute_hop_plot(G, is_original_graph=is_original_graph)
        s5_time = time.time() - s5_start
        print(f"       ✓ S5 completed in {s5_time:.1f}s (cumulative: {time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"     S5 failed: {e}")
        properties['hop_plot'] = np.array([0])
    
    try:
        print("     Computing S6: Hop-plot on largest component...")
        properties['hop_plot_wcc'] = compute_hop_plot_largest_wcc(G, is_original_graph=is_original_graph)
        print(f"       ✓ S6 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"     S6 failed: {e}")
        properties['hop_plot_wcc'] = np.array([0])
    
    try:
        print("     Computing S7: Singular vector...")
        properties['singular_vector'] = compute_singular_vector_distribution(G, is_original_graph=is_original_graph)
        print(f"       ✓ S7 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"     S7 failed: {e}")
        properties['singular_vector'] = np.array([0])
    
    try:
        print("     Computing S8: Singular values...")
        properties['singular_values'] = compute_singular_values_distribution(G, is_original_graph=is_original_graph)
        print(f"       ✓ S8 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"    S8 failed: {e}")
        properties['singular_values'] = np.array([0])
    
    try:
        print("      Computing S9: Clustering coefficient...")
        properties['clustering'] = compute_clustering_coefficient_distribution(G)
        print(f"       ✓ S9 completed ({time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"    S9 failed: {e}")
        properties['clustering'] = np.array([0])
    
    total_time = time.time() - start_time
    print(f"  ✅ All properties computed in {total_time:.2f}s")
    
    return properties

def compute_ks_statistic(dist1, dist2):
    """Compute Kolmogorov-Smirnov D-statistic between two distributions"""
    if len(dist1) == 0 or len(dist2) == 0:
        return 1.0  # Maximum dissimilarity
    
    try:
        # Normalize distributions to create empirical CDFs
        dist1_norm = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
        dist2_norm = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
        
        statistic, _ = ks_2samp(dist1_norm, dist2_norm)
        return statistic
    except:
        return 1.0

def perform_fireforest_analysis(original_G, providers, sample_sizes=[50, 100, 200], num_samples=100):
    """
    Perform comprehensive fireforest sampling analysis
    """
    import time
    analysis_start = time.time()
    
    print(f" Starting fireforest analysis of graph with {original_G.number_of_nodes()} nodes and {original_G.number_of_edges()} edges")
    print(f" Will analyze {len(sample_sizes)} sample sizes with {num_samples} samples each = {len(sample_sizes) * num_samples} total samples")
    
    # Compute properties of original graph (NO SAMPLING RESTRICTIONS)
    print("\n Computing properties of original graph...")
    original_properties = compute_all_graph_properties(original_G, is_original_graph=True)
    print(f" Original graph analysis completed!")
    
    results = {}
    
    for sample_size in sample_sizes:
        print(f"\nAnalyzing samples of size {sample_size}...")
        sample_results = {
            'in_degree': [],
            'out_degree': [],
            'wcc': [],
            'scc': [],
            'hop_plot': [],
            'hop_plot_wcc': [],
            'singular_vector': [],
            'singular_values': [],
            'clustering': []
        }
        
        # Generate samples and compute their properties
        sample_start = time.time()
        for i in range(num_samples):
            if i % 10 == 0:
                elapsed = time.time() - sample_start
                if i > 0:
                    rate = i / elapsed
                    eta = (num_samples - i) / rate if rate > 0 else 0
                    print(f"   Processing sample {i+1}/{num_samples} (Rate: {rate:.1f} samples/sec, ETA: {eta:.0f}s)")
                else:
                    print(f"   Processing sample {i+1}/{num_samples}")
            
            try:
                # Generate fireforest sample
                local_heads_number = min(5, len(providers))  # Adjust as needed
                sampled_nodes = fireforest_sample(original_G, sample_size, providers, local_heads_number, p=0.3)
                
                if len(sampled_nodes) < sample_size:
                    print(f"    Warning: Only got {len(sampled_nodes)} nodes instead of {sample_size}")
                
                # Create subgraph
                sample_G = original_G.subgraph(sampled_nodes)
                
                # Compute properties (with sampling restrictions for large subgraphs)
                sample_properties = compute_all_graph_properties(sample_G, is_original_graph=False)
                
                # Compute KS statistics
                for prop_name in sample_results.keys():
                    if prop_name in original_properties and prop_name in sample_properties:
                        ks_stat = compute_ks_statistic(original_properties[prop_name], 
                                                     sample_properties[prop_name])
                        sample_results[prop_name].append(ks_stat)
                    else:
                        sample_results[prop_name].append(1.0)  # Maximum dissimilarity
                        
            except Exception as e:
                print(f"    Error in sample {i+1}: {e}")
                # Add maximum dissimilarity for failed samples
                for prop_name in sample_results.keys():
                    sample_results[prop_name].append(1.0)
        
        # Calculate statistics for this sample size
        sample_time = time.time() - sample_start
        print(f"  ✅ Completed {num_samples} samples of size {sample_size} in {sample_time:.1f}s")
        
        results[sample_size] = {}
        for prop_name, ks_values in sample_results.items():
            results[sample_size][prop_name] = {
                'mean': np.mean(ks_values),
                'std': np.std(ks_values),
                'min': np.min(ks_values),
                'max': np.max(ks_values),
                'median': np.median(ks_values),
                'values': ks_values
            }
        
        # Show brief statistics for this sample size
        avg_ks = np.mean([results[sample_size][prop]['mean'] for prop in sample_results.keys()])
        print(f"      Average KS D-statistic: {avg_ks:.4f}")
    
    total_analysis_time = time.time() - analysis_start
    print(f"\n Fireforest analysis completed in {total_analysis_time/60:.1f} minutes!")
    
    return results

def print_analysis_summary(results):
    """Print summary of analysis results"""
    print("\n" + "="*80)
    print("FIREFOREST SAMPLING ANALYSIS SUMMARY")
    print("="*80)
    
    property_names = {
        'in_degree': 'S1: In-degree Distribution',
        'out_degree': 'S2: Out-degree Distribution', 
        'wcc': 'S3: Weakly Connected Components',
        'scc': 'S4: Strongly Connected Components',
        'hop_plot': 'S5: Hop-plot',
        'hop_plot_wcc': 'S6: Hop-plot on Largest WCC',
        'singular_vector': 'S7: First Singular Vector',
        'singular_values': 'S8: Singular Values',
        'clustering': 'S9: Clustering Coefficient'
    }
    
    for sample_size in sorted(results.keys()):
        print(f"\nSample Size: {sample_size}")
        print("-" * 40)
        
        for prop_key, prop_name in property_names.items():
            if prop_key in results[sample_size]:
                stats = results[sample_size][prop_key]
                print(f"{prop_name}:")
                print(f"  Mean KS D-statistic: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Median: {stats['median']:.4f}")
        print()

# Main Analysis Execution
if __name__ == "__main__":
    try:
        print("Loading Lightning Network graph data...")
        
        # Graph and providers should already be loaded from the data loading section above
        
        print(f" Graph loaded: {undirected_G.number_of_nodes()} nodes, {undirected_G.number_of_edges()} edges")
        print(f" Number of providers: {len(providers)}")
        
        # Perform the analysis
        print("\n" + "="*60)
        print(" STARTING FIREFOREST SAMPLING ANALYSIS")
        print("="*60)
        
        results = perform_fireforest_analysis(undirected_G, providers, 
                                            sample_sizes=[50, 100, 200], 
                                            num_samples=100)
        
        # Print summary
        print_analysis_summary(results)
        
        # Save results
        import pickle
        print("\n Saving results...")
        with open('fireforest_analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(" Results saved to 'fireforest_analysis_results.pkl'")
        
        print("\n" + "="*60)
        print(" ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
