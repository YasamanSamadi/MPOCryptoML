import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

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

def calculate_node_results(graph):
    result_dict = {}
    for node in graph.nodes():
        in_edges = graph.in_edges(node, data=True)
        out_edges = graph.out_edges(node, data=True)
        in_sum = sum([edge[2]['weight'] for edge in in_edges])
        out_sum = sum([edge[2]['weight'] for edge in out_edges])
        result = abs(out_sum - in_sum)
        result_dict[node] = result
    return result_dict

def normalize_results(result_dict):
    min_result = min(result_dict.values())
    max_result = max(result_dict.values())
    normalized_results = {node: (result - min_result) / (max_result - min_result) for node, result in result_dict.items()}
    epsilon = 1e-8
    normalized_results = {node: min(max(result, epsilon), 1 - epsilon) for node, result in normalized_results.items()}
    return normalized_results

def draw_graph(graph, normalized_results):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrowsize=20)
    result_labels = {node: f"{node}\nResult: {result:.3f}" for node, result in normalized_results.items()}
    nx.draw_networkx_labels(graph, pos, labels=result_labels)
    plt.show()

