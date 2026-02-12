# network_analysis.py

import networkx as nx
import matplotlib.pyplot as plt
import random

# --- Generate a Social Network ---
def generate_network(num_nodes=20, prob_edge=0.2):
    G = nx.erdos_renyi_graph(num_nodes, prob_edge)
    return G

# --- Plot Graph with Centrality Highlighted ---
def plot_graph(G):
    centrality = nx.degree_centrality(G)
    node_size = [5000 * centrality[n] for n in G.nodes()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='lightblue', edge_color='gray')
    plt.title("Social Network with Degree Centrality Highlighted")
    plt.show()

# --- Community Detection ---
def detect_communities(G):
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    print(f"Detected {len(communities)} communities.")
    return communities

# --- Monte Carlo Simulation: Influence Spread ---
def monte_carlo_spread(G, initial_node, num_simulations=1000):
    total_spread = 0
    for _ in range(num_simulations):
        visited = set()
        queue = [initial_node]
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                # Each neighbor has a 50% chance to be influenced
                neighbors = [n for n in G.neighbors(current) if random.random() < 0.5]
                queue.extend(neighbors)
        total_spread += len(visited)
    avg_spread = total_spread / num_simulations
    print(f"Avg spread from node {initial_node}: {avg_spread:.2f} nodes")
    return avg_spread

# --- Main Execution ---
if __name__ == "__main__":
    G = generate_network(num_nodes=25, prob_edge=0.15)
    plot_graph(G)

    communities = detect_communities(G)
    for i, c in enumerate(communities):
        print(f"Community {i + 1}: {sorted(c)}")

    # Run Monte Carlo Simulation from a random node
    start_node = random.choice(list(G.nodes()))
    monte_carlo_spread(G, start_node)
