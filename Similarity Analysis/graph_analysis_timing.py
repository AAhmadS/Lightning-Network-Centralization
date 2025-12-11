import time
import random
import pandas as pd
import networkx as nx
import numpy as np
import json
from scipy.stats import ks_2samp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import zstandard

def get_random_provider(providers, number_of_heads):
    """Get random providers for fireforest sampling"""
    random.seed()
    return random.sample(providers, min(number_of_heads, len(providers)))

def fireforest_sample(G, sample_size, providers, local_heads_number, p=0.3):
    """
    Performs a fire forest sampling algorithm to select a sample of nodes from the given graph `G`.
    """
    sampled_nodes = set()
    max_attempts = 5
    attempt = 0
    
    while len(sampled_nodes) < sample_size and attempt < max_attempts:
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

        # Check connectivity and size        
        if len(sampled_nodes) >= sample_size:
            break
        elif len(sampled_nodes) < sample_size:
            sampled_nodes = set()
            attempt += 1

    return sorted(list(sampled_nodes))

def is_subgraph_connected(G, nodes):
    """Check if subgraph is connected"""
    if len(nodes) <= 1:
        return True
    H = G.subgraph(nodes)
    return nx.is_connected(H)

def load_lngraph_data():
    """Load the Lightning Network graph from the compressed JSON file"""
    file_path = 'lngraph_2021_11_25__15_02.json.zst'
    
    print("Loading compressed Lightning Network data...")
    start_time = time.time()
    
    with open(file_path, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = json.load(reader)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        G.add_node(node['pub_key'], **{k: v for k, v in node.items() if k != 'pub_key'})
    
    # Add edges (both directions for Lightning channels)
    for edge in data.get('edges', []):
        G.add_edge(edge['node1_pub'], edge['node2_pub'], 
                  **{k: v for k, v in edge.items() if k not in ['node1_pub', 'node2_pub']})
        G.add_edge(edge['node2_pub'], edge['node1_pub'], 
                  **{k: v for k, v in edge.items() if k not in ['node1_pub', 'node2_pub']})
    
    # Convert to undirected for main analysis
    undirected_G = G.to_undirected()
    
    print(f"Graph created: {undirected_G.number_of_nodes()} nodes, {undirected_G.number_of_edges()} edges")
    return G, undirected_G

def analyze_graph_properties(G):
    """Analyze basic properties of the graph"""
    print("\n" + "="*60)
    print("GRAPH BASIC PROPERTIES ANALYSIS")
    print("="*60)
    
    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"Number of nodes: {n_nodes:,}")
    print(f"Number of edges: {n_edges:,}")
    print(f"Graph density: {nx.density(G):.6f}")
    print(f"Is connected: {nx.is_connected(G)}")
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    print(f"Degree std: {np.std(degrees):.2f}")
    
    # Connected components
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    if len(components) > 1:
        component_sizes = [len(c) for c in components]
        print(f"Largest component size: {max(component_sizes)} ({max(component_sizes)/n_nodes*100:.1f}%)")
        print(f"Smallest component size: {min(component_sizes)}")
    
    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'density': nx.density(G),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'components': len(components)
    }

def time_property_computation(G, sample_sizes=[50, 100, 200]):
    """Time each graph property computation using fireforest sampling"""
    print("\n" + "="*60)
    print("TIMING ANALYSIS FOR GRAPH PROPERTIES (with Fireforest Sampling)")
    print("="*60)
    
    # Create providers list (use high-degree nodes as providers)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    providers = [node for node, degree in sorted_nodes[:min(500, len(sorted_nodes))]]  # Top degree nodes as providers
    
    print(f"Using {len(providers)} providers for fireforest sampling")
    
    # Create sample graphs using fireforest sampling
    sample_graphs = {}
    sampling_times = {}
    
    for size in sample_sizes:
        print(f"\nGenerating fireforest sample of size {size}...")
        start_time = time.time()
        
        local_heads_number = min(10, len(providers))
        sample_nodes = fireforest_sample(G, size, providers, local_heads_number, p=0.3)
        
        sampling_time = time.time() - start_time
        sampling_times[size] = sampling_time
        
        if len(sample_nodes) > 0:
            sample_graphs[size] = G.subgraph(sample_nodes)
            print(f"Sample graph {size}: {len(sample_nodes)} nodes, {sample_graphs[size].number_of_edges()} edges")
            print(f"Sampling time: {sampling_time:.4f}s")
            print(f"Sample density: {nx.density(sample_graphs[size]):.6f}")
            print(f"Sample connected: {nx.is_connected(sample_graphs[size])}")
        else:
            print(f"Failed to generate sample of size {size}")
            # Create a simple connected sample as fallback
            seed_node = random.choice(list(G.nodes()))
            bfs_nodes = list(nx.bfs_tree(G, seed_node, depth_limit=3).nodes())[:size]
            sample_graphs[size] = G.subgraph(bfs_nodes)
            print(f"Fallback BFS sample: {len(bfs_nodes)} nodes, {sample_graphs[size].number_of_edges()} edges")
    
    # Test on full graph first (with limits for expensive operations)
    print(f"\nTiming on full graph ({G.number_of_nodes()} nodes):")
    full_graph_times = time_all_properties(G, is_full_graph=True)
    
    # Test on sample graphs
    sample_times = {}
    for size in sample_sizes:
        print(f"\nTiming on sample graph ({size} nodes):")
        sample_times[size] = time_all_properties(sample_graphs[size], is_full_graph=False)
    
    return full_graph_times, sample_times, sampling_times

def time_all_properties(G, is_full_graph=False):
    """Time computation of all graph properties"""
    times = {}
    
    # S1: In-degree distribution (convert to directed for this test)
    start_time = time.time()
    if G.is_directed():
        G_directed = G
    else:
        G_directed = G.to_directed()
    in_degrees = [G_directed.in_degree(node) for node in G_directed.nodes()]
    in_degree_dist = np.array(list(Counter(in_degrees).values()))
    times['S1_in_degree'] = time.time() - start_time
    print(f"  S1 (In-degree): {times['S1_in_degree']:.4f}s")
    
    # S2: Out-degree distribution
    start_time = time.time()
    out_degrees = [G_directed.out_degree(node) for node in G_directed.nodes()]
    out_degree_dist = np.array(list(Counter(out_degrees).values()))
    times['S2_out_degree'] = time.time() - start_time
    print(f"  S2 (Out-degree): {times['S2_out_degree']:.4f}s")
    
    # S3: Weakly connected components (use original undirected graph)
    start_time = time.time()
    if G.is_directed():
        wcc = list(nx.weakly_connected_components(G))
    else:
        wcc = list(nx.connected_components(G))
    wcc_sizes = [len(c) for c in wcc]
    times['S3_wcc'] = time.time() - start_time
    print(f"  S3 (WCC): {times['S3_wcc']:.4f}s")
    
    # S4: Strongly connected components
    start_time = time.time()
    if G.is_directed():
        scc = list(nx.strongly_connected_components(G))
    else:
        # For undirected graphs, strongly connected = connected
        scc = list(nx.connected_components(G))
    scc_sizes = [len(c) for c in scc]
    times['S4_scc'] = time.time() - start_time
    print(f"  S4 (SCC): {times['S4_scc']:.4f}s")
    
    # S5: Hop-plot (limited for large graphs)
    start_time = time.time()
    nodes = list(G.nodes())
    max_hops = 5  # Limit hops for timing
    
    if is_full_graph and len(nodes) > 1000:
        # Sample nodes for large graphs
        sample_nodes = random.sample(nodes, 100)
    else:
        sample_nodes = nodes[:min(100, len(nodes))]
    
    hop_counts = []
    for h in range(1, max_hops + 1):
        reachable_pairs = 0
        for node in sample_nodes:
            try:
                paths = nx.single_source_shortest_path_length(G, node, cutoff=h)
                reachable_pairs += len(paths) - 1
            except:
                continue
        hop_counts.append(reachable_pairs)
    times['S5_hop_plot'] = time.time() - start_time
    print(f"  S5 (Hop-plot): {times['S5_hop_plot']:.4f}s")
    
    # S6: Hop-plot on largest component
    start_time = time.time()
    if G.is_directed():
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_wcc = max(nx.connected_components(G), key=len)
    
    if len(largest_wcc) > 1000:
        largest_wcc = random.sample(list(largest_wcc), 1000)
    
    G_largest = G.subgraph(largest_wcc)
    # Simplified hop-plot computation
    sample_nodes_wcc = list(G_largest.nodes())[:min(50, len(G_largest.nodes()))]
    hop_counts_wcc = []
    for h in range(1, 4):  # Even more limited for timing
        reachable_pairs = 0
        for node in sample_nodes_wcc:
            try:
                paths = nx.single_source_shortest_path_length(G_largest, node, cutoff=h)
                reachable_pairs += len(paths) - 1
            except:
                continue
        hop_counts_wcc.append(reachable_pairs)
    times['S6_hop_plot_wcc'] = time.time() - start_time
    print(f"  S6 (Hop-plot WCC): {times['S6_hop_plot_wcc']:.4f}s")
    
    # S7: Singular vector (most expensive)
    start_time = time.time()
    try:
        A = nx.adjacency_matrix(G)
        if A.shape[0] > 500:  # Limit size for timing
            nodes_sample = random.sample(list(G.nodes()), min(500, A.shape[0]))
            G_sample = G.subgraph(nodes_sample)
            A = nx.adjacency_matrix(G_sample)
        
        # Check if matrix has any non-zero elementsس
        if A.nnz == 0:  # No edges in the graph
            singular_vector = np.zeros(A.shape[0])
        else:
            k = min(10, A.shape[0] - 1)  # Very limited for timing
            if k > 0 and A.shape[0] > 1:
                # Add small diagonal term to avoid zero matrix issues
                A_regularized = A + 1e-10 * np.eye(A.shape[0])
                u, s, vt = svds(A_regularized.astype(float), k=k)
                singular_vector = np.abs(u[:, 0])
            else:
                singular_vector = np.zeros(A.shape[0])
    except Exception as e:
        print(f"    SVD error: {e}")
        singular_vector = np.zeros(1)
    times['S7_singular_vector'] = time.time() - start_time
    print(f"  S7 (Singular vector): {times['S7_singular_vector']:.4f}s")
    
    # S8: Singular values (also expensive)
    start_time = time.time()
    try:
        # Reuse the computation from S7 if possible
        pass  # Already computed above
    except:
        pass
    times['S8_singular_values'] = time.time() - start_time
    print(f"  S8 (Singular values): {times['S8_singular_values']:.4f}s")
    
    # S9: Clustering coefficient
    start_time = time.time()
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    # Limit nodes for large graphs
    if is_full_graph and G_undirected.number_of_nodes() > 1000:
        nodes_sample = random.sample(list(G_undirected.nodes()), 1000)
        G_clustering = G_undirected.subgraph(nodes_sample)
    else:
        G_clustering = G_undirected
    
    clustering = nx.clustering(G_clustering)
    degrees = dict(G_clustering.degree())
    
    degree_clustering = defaultdict(list)
    for node in G_clustering.nodes():
        degree = degrees[node]
        degree_clustering[degree].append(clustering[node])
    
    times['S9_clustering'] = time.time() - start_time
    print(f"  S9 (Clustering): {times['S9_clustering']:.4f}s")
    
    return times

def estimate_total_time(full_times, sample_times, sampling_times, num_samples=100):
    """Estimate total computation time for the full analysis including sampling overhead"""
    print("\n" + "="*60)
    print("TIME ESTIMATION FOR FULL ANALYSIS (including Fireforest Sampling)")
    print("="*60)
    
    # Time for original graph (once)
    full_graph_time = sum(full_times.values())
    print(f"Time for original graph analysis: {full_graph_time:.2f} seconds")
    
    # Time for fireforest sampling
    total_sampling_time = 0
    for size in [50, 100, 200]:
        if size in sampling_times:
            sampling_time_for_size = sampling_times[size] * num_samples
            total_sampling_time += sampling_time_for_size
            print(f"Fireforest sampling time for {num_samples} samples of size {size}: {sampling_time_for_size:.2f} seconds")
    
    print(f"Total fireforest sampling time: {total_sampling_time:.2f} seconds")

    # Time per sample for each size (property computation)
    total_sample_time = 0
    for size in [50, 100, 200]:
        if size in sample_times:
            time_per_sample = sum(sample_times[size].values())
            total_time_for_size = time_per_sample * num_samples
            total_sample_time += total_time_for_size
            
            print(f"Property computation per sample (size {size}): {time_per_sample:.3f} seconds")
            print(f"Total property computation for {num_samples} samples of size {size}: {total_time_for_size/60:.1f} minutes")
    
    total_estimated_time = full_graph_time + total_sampling_time + total_sample_time
    
    print(f"\nTotal estimated time: {total_estimated_time/60:.1f} minutes ({total_estimated_time/3600:.1f} hours)")
    
    # Breakdown by property
    print(f"\nTime breakdown by property (for all samples):")
    for prop in ['S1_in_degree', 'S2_out_degree', 'S3_wcc', 'S4_scc', 'S5_hop_plot', 
                 'S6_hop_plot_wcc', 'S7_singular_vector', 'S8_singular_values', 'S9_clustering']:
        prop_time = full_times.get(prop, 0)
        for size in [50, 100, 200]:
            if size in sample_times:
                prop_time += sample_times[size].get(prop, 0) * num_samples
        
        print(f"  {prop}: {prop_time/60:.2f} minutes")
    
    return total_estimated_time

if __name__ == "__main__":
    try:
        # Load the Lightning Network graph
        directed_G, undirected_G = load_lngraph_data()
        
        # Analyze basic properties
        basic_props = analyze_graph_properties(undirected_G)
        
        # Time property computations
        full_times, sample_times, sampling_times = time_property_computation(undirected_G, sample_sizes=[50, 100, 200])
        
        # Estimate total analysis time
        estimated_time = estimate_total_time(full_times, sample_times, sampling_times, num_samples=100)
        
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        
        if estimated_time > 7200:  # More than 2 hours
            print("⚠️  Analysis will take a long time (>2 hours)")
            print("Consider reducing:")
            print("- Number of samples (currently 100)")
            print("- Sample sizes (currently 50, 100, 200)")
            print("- Hop-plot depth (currently limited)")
            print("- SVD dimensions (currently limited)")
        elif estimated_time > 1800:  # More than 30 minutes
            print("⏰ Analysis will take moderate time (30min - 2hours)")
            print("Consider running during a break or overnight")
        else:
            print("✅ Analysis should complete in reasonable time (<30 minutes)")
        
        print(f"\nGraph complexity indicators:")
        print(f"- Nodes: {basic_props['nodes']:,}")
        print(f"- Edges: {basic_props['edges']:,}")
        print(f"- Density: {basic_props['density']:.6f}")
        print(f"- Max degree: {basic_props['max_degree']}")
        
    except Exception as e:
        print(f"Error during timing analysis: {e}")
        import traceback
        traceback.print_exc()