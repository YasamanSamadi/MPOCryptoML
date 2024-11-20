import networkx as nx
import math
from datetime import datetime
import random

def read_graph_from_file(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            source, target, weight, timestamp_str = line.split()

            if len(timestamp_str) == 10:
                timestamp = datetime.utcfromtimestamp(int(timestamp_str))
            else:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')

            weight = int(weight)
            G.add_edge(int(source), int(target), weight=weight, timestamp=timestamp)
    return G

def initialize_values(graph, sources):
    residue_values = {(s, u): 0 if (s, u) != (s, s) else 1 for s in sources for u in graph.nodes}
    reserve_values = {(s, u): 0 for s in sources for u in graph.nodes}
    PPR_values = {(s, v): 0 for s in sources for v in graph.nodes}
    random_walks_numbers = {(s, u): 0 for s in sources for u in graph.nodes}
    return residue_values, reserve_values, PPR_values, random_walks_numbers

def calculate_k_s(graph, alpha, source, epsilon, pf):
    out_neighbors = list(graph.successors(source))
    number_of_children_of_source_node = len(out_neighbors)
    numerator = ((2 / 3) * epsilon + 2) * number_of_children_of_source_node * math.log(2 / pf)
    denominator = (epsilon ** 2) * alpha * (1 - alpha)
    k_s = round(numerator / denominator)  # Round the result
    return k_s

def random_walks(random_walks_numbers, source, alpha, residue_values, k_s, current_node, graph, sources):
    for child in graph.successors(current_node):
        edge_data = graph.get_edge_data(source, child)
        
        if edge_data is not None:
            edge_weight = edge_data.get('weight', 1)
            probability = edge_weight / sum(graph.get_edge_data(s, child).get('weight', 1) for s in sources if graph.get_edge_data(s, child) is not None)
            
            if random.random() < probability:
                random_walks_numbers[(source, child)] = (1 - alpha) * residue_values[(source, current_node)] * k_s
    return random_walks_numbers

def forward_push(graph, source, alpha, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s, sources):
    for current_node in graph.nodes:
        if current_node in sources:
            children_of_source_node = list(graph.successors(current_node))
            if not children_of_source_node:
                continue
            number_of_children_of_source_node = len(children_of_source_node)

            for child in children_of_source_node:
                residue_values[(source, child)] += (1 - alpha) * residue_values[(source, current_node)] / number_of_children_of_source_node

            reserve_values[(source, current_node)] += alpha * residue_values[(source, current_node)]

            random_walks_numbers = random_walks(random_walks_numbers, source, alpha, residue_values, k_s, current_node, graph, sources)

            if residue_values[(source, current_node)] > (number_of_children_of_source_node) / (alpha * k_s) and (number_of_children_of_source_node > 0):
                PPR_values[(source, current_node)] = reserve_values[(source, current_node)]
            else:
                PPR_values[(source, current_node)] = reserve_values[(source, current_node)] + (1 / k_s)

            residue_values[(source, current_node)] = 0

        else:
            children_of_current_node = list(graph.successors(current_node))
            if not children_of_current_node:
                continue
            number_of_children_of_current_node = len(children_of_current_node)

            for child in children_of_current_node:
                residue_values[(source, child)] += (1 - alpha) * residue_values[(source, current_node)] / number_of_children_of_current_node

            reserve_values[(source, current_node)] += alpha * residue_values[(source, current_node)]

            random_walks_numbers = random_walks(random_walks_numbers, source, alpha, residue_values, k_s, current_node, graph, sources)

            if residue_values[(source, current_node)] > (number_of_children_of_current_node) / (alpha * k_s) and (number_of_children_of_current_node > 0):
                PPR_values[(source, current_node)] = reserve_values[(source, current_node)]
            else:
                PPR_values[(source, current_node)] = reserve_values[(source, current_node)] + (1 / k_s)

            residue_values[(source, current_node)] = 0

    return PPR_values, residue_values, reserve_values


example_graph = nx.DiGraph()
example_graph.add_edge(1, 2, weight=3)
example_graph.add_edge(1, 3, weight=2)
example_graph.add_edge(2, 4, weight=1)
example_graph.add_edge(3, 4, weight=4)

example_sources = [3, 2]

graph = read_graph_from_file('dataset1.txt')

# Initialize values
residue_values, reserve_values, PPR_values, random_walks_numbers = initialize_values(graph, example_sources)

#parameters
alpha = 0.85
epsilon = 0.001
pf = 0.15

#k_s 
for source in example_sources:
    k_s = calculate_k_s(graph, alpha, source, epsilon, pf)
    print(f"k_s for source {source}: {k_s}")

for source in example_sources:
    PPR_values, residue_values, reserve_values = forward_push(graph, source, alpha, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s, example_sources)

for source in example_sources:
    print(f"\nPPR values for source {source}:")
    for node in graph.nodes:
        print(f"Node {node}: {PPR_values[(source, node)]}")
