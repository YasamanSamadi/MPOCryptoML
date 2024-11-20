import networkx as nx
import math
from datetime import datetime
import finding_source as source

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

def initialize_values(graph, source_nodes):
    residue_values = {(source, u): 0 if (source, u) != (source, source) else 1 for u in graph.nodes for source in source_nodes}
    reserve_values = {(source, u): 0 for u in graph.nodes for source in source_nodes}
    PPR_values = {(source, v): 0 for v in graph.nodes for source in source_nodes}
    random_walks_numbers = {(source, u): 0 for u in graph.nodes for source in source_nodes}
    return residue_values, reserve_values, PPR_values, random_walks_numbers

def calculate_k_s(graph, alpha, source, epsilon, pf):
    out_neighbors = list(graph.successors(source))
    number_of_children_of_source_node = len(out_neighbors)
    numerator = ((2 / 3) * epsilon + 2) * number_of_children_of_source_node * math.log(2 / pf)
    denominator = (epsilon ** 2) * alpha * (1 - alpha)
    k_s = round(numerator / denominator)  # Round the result
    return k_s

def random_walks(random_walks_numbers, source, alpha, residue_values, k_s):
    random_walks_numbers[current_node] = (1 - alpha) * residue_values[(source, current_node)] * k_s
    return random_walks_numbers

def forward_push(graph, source_nodes, alpha, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s):
    for source in source_nodes:
        for current_node in graph.nodes:
            if current_node == source:
                children_of_source_node = list(graph.successors(current_node))
                if not children_of_source_node:
                    continue
                number_of_children_of_source_node = len(children_of_source_node)

                for child in children_of_source_node:
                    residue_values[(source, child)] += (1 - alpha) * residue_values[(source, current_node)] / number_of_children_of_source_node

                reserve_values[(source, current_node)] += alpha * residue_values[(source, current_node)]

                print(random_walks(random_walks_numbers, source, alpha, residue_values, k_s, current_node))

                if residue_values[(source, current_node)] > (number_of_children_of_source_node) / (alpha * k_s) and (number_of_children_of_source_node > 0):
                    PPR_values[(source, current_node)] = reserve_values[(source, current_node)]
                else:
                    PPR_values[(source, current_node)] = reserve_values[(source, current_node)] + (1 / k_s)

                residue_values[(source, current_node)] = 0

                print(f"Iteration {current_node}: PPR Values:", PPR_values)
                print(f"Iteration {current_node}: Residue Values:", residue_values)
                print(f"Iteration {current_node}: Reserve Values:", reserve_values)

            else:
                children_of_current_node = list(graph.successors(current_node))
                if not children_of_current_node:
                    continue
                number_of_children_of_current_node = len(children_of_current_node)

                for child in children_of_current_node:
                    residue_values[(source, child)] += (1 - alpha) * residue_values[(source, current_node)] / number_of_children_of_current_node

                reserve_values[(source, current_node)] += alpha * residue_values[(source, current_node)]

                print(random_walks(random_walks_numbers, source, alpha, residue_values, k_s, current_node))

                if residue_values[(source, current_node)] > (number_of_children_of_source_node) / (alpha * k_s) and (number_of_children_of_source_node > 0):
                    PPR_values[(source, current_node)] = reserve_values[(source, current_node)]
                else:
                    PPR_values[(source, current_node)] = reserve_values[(source, current_node)] + (1 / k_s)

                residue_values[(source, current_node)] = 0

                print(f"Iteration {current_node}: PPR Values:", PPR_values)
                print(f"Iteration {current_node}: Residue Values:", residue_values)
                print(f"Iteration {current_node}: Reserve Values:", reserve_values)

    return PPR_values, residue_values, reserve_values

# Example usage:
file_path = '0.1_test.txt'
graph = read_graph_from_file(file_path)

# Example: Assuming `find_source_nodes` is a function that returns the list of source nodes
source_nodes = source(graph)

alpha = 0.85
epsilon = 0.0001
pf = 1

residue_values, reserve_values, PPR_values, random_walks_numbers = initialize_values(graph, source_nodes)

for source in source_nodes:
    k_s = calculate_k_s(graph, alpha, source, epsilon, pf)
    PPR_values, residue_values, reserve_values = forward_push(graph, [source], alpha, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s)

print("Final PPR Values:")
print(PPR_values)
