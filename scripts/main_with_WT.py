import networkx as nx
import pandas as pd
from datetime import datetime
import ppr as ppr
import advanced_weight as weight
import advanced_scoring_timestamp as time
import AnomalyScore as ascore
from WT import find_anomalous_nodes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from opt_WDDS import find_weighted_densest_subgraph

# Constants
ALPHA = 0.5
EPSILON = 0.5
FILE_PATH = '0.1_test.txt'
OUT_PUT_FILE_WEIGHT = '0.1_weight.txt'
OUT_PUT_FILE_TIMESTAMP = '0.1_timestamp.txt'
OUT_PUT_FILE_ANOMALY_SCORES = "0.1_anomalyscore.txt"
OUT_PUT_FILE_MAX_ANOMALY_SCORE = "0.1_Max_anomalyscore.txt"
OUT_PUT_FILE_WT_ANOMALY_NODES = "0.1_WT_anomaly_nodes.txt"
OUT_PUT_FILE_ANOMALY_NODES = "0.1_anomalous_nodes.txt"  
 

# Create Graph based on input data
def read_graph_from_file(file_path):
    G = nx.DiGraph()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                source, target, edge_weight, timestamp_str, anomaly_label = line.split()
                if len(timestamp_str) == 10:
                    timestamp = datetime.utcfromtimestamp(int(timestamp_str))
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                edge_weight = int(edge_weight)
                anomaly_label = int(anomaly_label)  
                G.add_edge(int(source), int(target), weight=edge_weight, timestamp=timestamp, anomaly_label=anomaly_label)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        raise
    except Exception as e:
        print(f"Error reading graph from file: {str(e)}")
        raise
    return G

# Example usage
try:
    G = read_graph_from_file(FILE_PATH)
except Exception as e:
    print(f"Error: {str(e)}")

# WDDS alg
densest_subgraph = find_weighted_densest_subgraph(G)

# PPR alg
pf = 1
#source_node = [1, 2, 3]
source_node = [1]

residue_values, reserve_values, PPR_values, random_walks_numbers = ppr.initialize_values_for_multiple_sources(G, source_node)
k_s = ppr.calculate_k_s_for_multiple_sources(G, ALPHA, source_node, EPSILON, pf)
PPR_values, residue_values, reserve_values = ppr.forward_push_for_multiple_sources(G, source_node, ALPHA, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s)

df_ppr = pd.DataFrame(list(PPR_values.items()), columns=['Edge', 'PPR'])
df_ppr[['source', 'target']] = pd.DataFrame(df_ppr['Edge'].tolist(), index=df_ppr.index)
df_ppr = df_ppr.drop('Edge', axis=1)
df_ppr = df_ppr[['target', 'PPR']]
df_ppr = df_ppr.rename(columns={'target': 'Node'})
print("PPR RESULTS")
print(df_ppr)

# Weight alg
result_dict = weight.calculate_node_results(G)
normalized_results = weight.normalize_results(result_dict)
weight_normalized_df = pd.DataFrame(list(normalized_results.items()), columns=['Node', 'Normalized_Result'])
weight_normalized_df.to_csv(OUT_PUT_FILE_WEIGHT, sep='\t', index=False)

# Timestamp alg
try:
    time.print_edge_timestamps(G)
    nodes_with_sufficient_data, absolute_differences = time.analyze_timestamps(G, G.nodes)

    if len(nodes_with_sufficient_data) > 0:
        sorted_data = time.normalize_differences(absolute_differences, nodes_with_sufficient_data)
        df_sorted_diff = pd.DataFrame(sorted_data, columns=['Node', 'Sorted_Normalized_Difference'])
        df_sorted_diff = df_sorted_diff.set_index('Node')
        all_nodes = pd.DataFrame({'Node': G.nodes})
        time_df_result = pd.merge(all_nodes, df_sorted_diff, how='left', left_on='Node', right_index=True)
        time_df_result['Sorted_Normalized_Difference'].fillna(0, inplace=True)
        print(time_df_result)
        time_df_result.to_csv(OUT_PUT_FILE_TIMESTAMP, sep='\t', index=False)
    else:
        print("\nNo nodes with sufficient data for normalization.")

except ValueError as e:
    print(f"Error: {str(e)}")

# Merge DataFrames
final_df = pd.merge(df_ppr, weight_normalized_df, on='Node', how='outer')
final_df = pd.merge(final_df, time_df_result, on='Node', how='outer')
final_df = final_df.rename(columns={'Normalized_Result': 'Weight', 'Sorted_Normalized_Difference': 'Timestamp'})

# Find nodes meeting conditions using the new module
filtered_nodes_df = find_anomalous_nodes(final_df)

# Print or save the result 
print("\nNodes meeting conditions:")
print(filtered_nodes_df)
filtered_nodes_df.to_csv(OUT_PUT_FILE_WT_ANOMALY_NODES, sep='\t', index=False)  # Save to file

# Anomaly Score
anomaly_scores, (max_anomaly_node, max_anomaly_score) = ascore.calculate_anomaly_scores(final_df)

# Print or save the result 
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
print(f"Nodes meeting conditions have been saved to {OUT_PUT_FILE_WT_ANOMALY_NODES}")
