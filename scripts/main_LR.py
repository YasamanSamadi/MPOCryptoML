import networkx as nx
import pandas as pd
from datetime import datetime
import ppr as ppr
import advanced_weight as weight
import advanced_scoring_timestamp as time
import AnomalyScore as ascore

# Constants
ALPHA = 0.5
EPSILON = 0.5
FILE_PATH = 'dataset1.csv'
OUT_PUT_FILE_WEIGHT = 'weight.txt'
OUT_PUT_FILE_TIMESTAMP = 'timestamp.txt'
OUT_PUT_FILE_ANOMALY_SCORES = "anomalyscore.txt"
OUT_PUT_FILE_MAX_ANOMALY_SCORE = "Max_anomalyscore.txt"

# Function to identify source nodes dynamically
def find_source_nodes(graph):
    # Nodes with no incoming edges (in-degree = 0) and outgoing edges (out-degree > 0)
    source_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
    
    # If no such nodes exist, fallback to nodes with the highest out-degree
    if not source_nodes:
        max_out_degree = max(dict(graph.out_degree).values())
        source_nodes = [node for node, degree in graph.out_degree if degree == max_out_degree]
    
    return source_nodes

# Function to read graph from CSV file
def read_graph_from_csv(file_path):
    G = nx.DiGraph()
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            edge_weight = float(row['weight'])
            timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
            G.add_edge(source, target, weight=edge_weight, timestamp=timestamp)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        raise
    except Exception as e:
        print(f"Error reading graph from file: {str(e)}")
        raise
    return G

# Load graph
try:
    G = read_graph_from_csv(FILE_PATH)
except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Identify source nodes
source_nodes = find_source_nodes(G)
print("Source nodes identified:", source_nodes)

# Personalized PageRank analysis
pf = 1
residue_values, reserve_values, PPR_values, random_walks_numbers = ppr.initialize_values_for_multiple_sources(G, source_nodes)
k_s = ppr.calculate_k_s_for_multiple_sources(G, ALPHA, source_nodes, EPSILON, pf)
PPR_values, residue_values, reserve_values = ppr.forward_push_for_multiple_sources(G, source_nodes, ALPHA, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s)

# Convert PPR results to DataFrame
df_ppr = pd.DataFrame(list(PPR_values.items()), columns=['Edge', 'PPR'])
df_ppr[['source', 'target']] = pd.DataFrame(df_ppr['Edge'].tolist(), index=df_ppr.index)
df_ppr = df_ppr.drop('Edge', axis=1)[['target', 'PPR']].rename(columns={'target': 'Node'})
print("PPR RESULTS")    
print(df_ppr)

# Weight normalization
result_dict = weight.calculate_node_results(G)
normalized_results = weight.normalize_results(result_dict)
weight_normalized_df = pd.DataFrame(list(normalized_results.items()), columns=['Node', 'Normalized_Result'])
weight_normalized_df.to_csv(OUT_PUT_FILE_WEIGHT, sep='\t', index=False)

# Timestamp analysis
try:
    time.print_edge_timestamps(G)
    nodes_with_sufficient_data, absolute_differences = time.analyze_timestamps(G, G.nodes)

    if nodes_with_sufficient_data:
        sorted_data = time.normalize_differences(absolute_differences, nodes_with_sufficient_data)
        df_sorted_diff = pd.DataFrame(sorted_data, columns=['Node', 'Sorted_Normalized_Difference']).set_index('Node')

        all_nodes = pd.DataFrame({'Node': G.nodes})
        time_df_result = pd.merge(all_nodes, df_sorted_diff, how='left', left_on='Node', right_index=True)
        time_df_result['Sorted_Normalized_Difference'].fillna(0, inplace=True)
        time_df_result.to_csv(OUT_PUT_FILE_TIMESTAMP, sep='\t', index=False)
        print(time_df_result)
    else:
        print("\nNo nodes with sufficient data for normalization.")
except ValueError as e:
    print(f"Error: {str(e)}")

# Merge results
final_df = pd.merge(df_ppr, weight_normalized_df, on='Node', how='outer')
final_df = pd.merge(final_df, time_df_result, on='Node', how='outer')
final_df = final_df.rename(columns={'Normalized_Result': 'Weight', 'Sorted_Normalized_Difference': 'Timestamp'})
print(final_df)

# Anomaly score computation
anomaly_scores, (max_anomaly_node, max_anomaly_score) = ascore.calculate_anomaly_scores(final_df)

# Save anomaly scores to file
with open(OUT_PUT_FILE_ANOMALY_SCORES, 'w') as output_file:
    output_file.write("Node\tAnomaly Score\n")
    for node, score in anomaly_scores:
        output_file.write(f"{node}\t{score}\n")
    output_file.write("\nMaximum Anomaly Score:\n")
    output_file.write(f"Node: {max_anomaly_node}\n")
    output_file.write(f"Score: {max_anomaly_score}\n")

with open(OUT_PUT_FILE_MAX_ANOMALY_SCORE, 'w') as max_anomaly_file:
    max_anomaly_file.write(f"Node: {max_anomaly_node}\n")
    max_anomaly_file.write(f"Score: {max_anomaly_score}\n")

print(f"Output has been saved to {OUT_PUT_FILE_ANOMALY_SCORES}")
print(f"Maximum anomaly score information has been saved to {OUT_PUT_FILE_MAX_ANOMALY_SCORE}")
