import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

def read_graph_from_file(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            source, target, weight, timestamp_str = line.split()

            # Check the length of the timestamp string to determine the format
            if len(timestamp_str) == 10:
                timestamp = datetime.utcfromtimestamp(int(timestamp_str))
            else:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')

            weight = int(weight)
            G.add_edge(int(source), int(target), weight=weight, timestamp=timestamp)
    return G

def print_edge_timestamps(G):
    """Print timestamps for each edge in the graph."""
    print("Edge Timestamps:")
    for source, target, timestamp in G.edges(data='timestamp'):
        print(f"{source} -> {target}: {timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}")

def analyze_timestamps(G, nodes):
    """Perform timestamp analysis for each node in the graph."""
    absolute_differences = []
    nodes_with_sufficient_data = []

    for node in nodes:
        indegree_timestamps = [G.edges[prev_node, node]['timestamp'] for prev_node in G.predecessors(node)]
        outdegree_timestamps = [G.edges[node, next_node]['timestamp'] for next_node in G.successors(node)]

        if indegree_timestamps and outdegree_timestamps:
            nodes_with_sufficient_data.append(node)

            indegree_time_diff = [indegree_timestamps[i] - indegree_timestamps[i - 1] for i in range(1, len(indegree_timestamps))]
            outdegree_time_diff = [outdegree_timestamps[i] - outdegree_timestamps[i - 1] for i in range(1, len(outdegree_timestamps))]

            indegree_std_dev = sum((time_diff.total_seconds() for time_diff in indegree_time_diff)) / len(indegree_time_diff) if len(indegree_time_diff) > 0 else 0
            outdegree_std_dev = sum((time_diff.total_seconds() for time_diff in outdegree_time_diff)) / len(outdegree_time_diff) if len(outdegree_time_diff) > 0 else 0

            absolute_difference = abs(indegree_std_dev - outdegree_std_dev)
            absolute_differences.append(absolute_difference)

            print(f"\nNode '{node}' Timestamp Analysis:")
            print(f"Indegree Timestamps: {indegree_timestamps}")
            print(f"Indegree Standard Deviation: {indegree_std_dev}")
            print(f"Outdegree Timestamps: {outdegree_timestamps}")
            print(f"Outdegree Standard Deviation: {outdegree_std_dev}")
            print(f"Absolute Difference: {absolute_difference}")
        else:
            print(f"\nNode '{node}' has insufficient data for analysis.")

    return nodes_with_sufficient_data, absolute_differences


def normalize_differences(absolute_differences, nodes_with_sufficient_data):
    """Normalize absolute differences to decimal digits between 0 and 1."""
    if len(absolute_differences) > 0:
        min_absolute_difference = min(absolute_differences)
        max_absolute_difference = max(absolute_differences)

        if max_absolute_difference == min_absolute_difference:
            normalized_differences = [0.5] * len(absolute_differences)
        else:
            normalized_differences = [(d - min_absolute_difference) / (max_absolute_difference - min_absolute_difference) for d in absolute_differences]

        normalized_differences = [0.0001 if x == 0 else 0.9999 if x == 1 else x for x in normalized_differences]
        
        return sorted(zip(nodes_with_sufficient_data, normalized_differences), key=lambda x: x[1])
    else:
        return []


def visualize_graph(G):
    """Visualize the graph."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', arrowsize=20)
    edge_labels = {(source, target): timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") for source, target, timestamp in G.edges(data='timestamp')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

# Usage
#file_path = 'dataset1.txt'
#G = read_graph_from_file(file_path)

#try:
    #print_edge_timestamps(G)
    #nodes_with_sufficient_data, absolute_differences = analyze_timestamps(G, G.nodes)

    #if len(nodes_with_sufficient_data) > 0:
     #   sorted_data = normalize_differences(absolute_differences, nodes_with_sufficient_data)
      #  print("\nSorted Normalized Differences:")
       # for node, normalized_diff in sorted_data:
        #    print(f"Node '{node}': {normalized_diff}")
    #else:
     #   print("\nNo nodes with sufficient data for normalization.")

    #visualize_graph(G)

#except ValueError as e:
    #print(f"Error: {str(e)}")
