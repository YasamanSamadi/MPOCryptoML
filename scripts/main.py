import networkx as nx
import pandas as pd
from datetime import datetime

#import advanced_ppr as ppr
import ppr as ppr
import advanced_weight as weight
#import advanced_timestamp as time
import advanced_scoring_timestamp as time
import AnomalyScore as ascore


# Constants
ALPHA = 0.5
EPSILON = 0.5
FILE_PATH = 'dataset1.txt'


#OUTPUT
OUT_PUT_FILE_WEIGHT = 'weight.txt'
OUT_PUT_FILE_TIMESTAMP = 'timestamp.txt'
OUT_PUT_FILE_ANOMALY_SCORES = "anomalyscore.txt"
OUT_PUT_FILE_MAX_ANOMALY_SCORE = "Max_anomalyscore.txt"

########### CREATE GRAPH BASED ON INPUT DATA ########################
def read_graph_from_file(file_path):
    G = nx.DiGraph()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                source, target, edge_weight, timestamp_str = line.split()

    
                if len(timestamp_str) == 10:
                    timestamp = datetime.utcfromtimestamp(int(timestamp_str))
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')

                edge_weight = int(edge_weight)
                G.add_edge(int(source), int(target), weight=edge_weight, timestamp=timestamp)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        raise
    except Exception as e:
        print(f"Error reading graph from file: {str(e)}")
        raise
    return G


try:
    G = read_graph_from_file(FILE_PATH)
    
except Exception as e:
    
    print(f"Error: {str(e)}")
#G = read_graph_from_file(FILE_PATH)

############################# PPR VALUE ########################
# Parameters for personalized PageRank
pf = 1
source_node = 1, 2, 3  # Assuming source node is 1

# Initialize values
#residue_values, reserve_values, PPR_values, random_walks_numbers = ppr.initialize_values(G, source_node)
residue_values, reserve_values, PPR_values, random_walks_numbers = ppr.initialize_values_for_multiple_sources(G, source_node)

# Calculate K for the source
#k_s = ppr.calculate_k_s(G, ALPHA, source_node, EPSILON, pf)
k_s = ppr.calculate_k_s_for_multiple_sources(G, ALPHA, source_node, EPSILON, pf)

# Perform iterations (you can loop this for multiple iterations)
#PPR_values, residue_values, reserve_values = ppr.forward_push(G, source_node, ALPHA, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s)
PPR_values, residue_values, reserve_values = ppr.forward_push_for_multiple_sources(G, source_node, ALPHA, residue_values, reserve_values, PPR_values, random_walks_numbers, k_s)

# Convert the final result dictionary to a DataFrame
df_ppr = pd.DataFrame(list(PPR_values.items()), columns=['Edge', 'PPR'])

# Split the 'Edge' column into separate 'source' and 'target' columns
df_ppr[['source', 'target']] = pd.DataFrame(df_ppr['Edge'].tolist(), index=df_ppr.index)

# Drop the original 'Edge' column
df_ppr = df_ppr.drop('Edge', axis=1)

# Reorder the columns as needed
df_ppr = df_ppr[['target', 'PPR']]
df_ppr = df_ppr.rename(columns={'target': 'Node'})
# Print the resulting DataFrame
print("PPR RESULTS")    
print(df_ppr)

###################################### WEIGHT ########################
# Calculate node results
result_dict = weight.calculate_node_results(G)

# Normalize the results
normalized_results = weight.normalize_results(result_dict)
weight_normalized_df = pd.DataFrame(list(normalized_results.items()), columns=['Node', 'Normalized_Result'])

# Print the normalized results DataFrame
#print("\nNormalized Results:")
#print(weight_normalized_df)
weight_normalized_df.to_csv(OUT_PUT_FILE_WEIGHT, sep='\t', index=False)

################################################## TIMESTAMP ####################
try:
    time.print_edge_timestamps(G)
    nodes_with_sufficient_data, absolute_differences = time.analyze_timestamps(G, G.nodes)

    if len(nodes_with_sufficient_data) > 0:
        sorted_data = time.normalize_differences(absolute_differences, nodes_with_sufficient_data)

        # Create a DataFrame for Sorted Normalized Differences
        df_sorted_diff = pd.DataFrame(sorted_data, columns=['Node', 'Sorted_Normalized_Difference'])
        df_sorted_diff = df_sorted_diff.set_index('Node')

        # Merge with nodes that are not in the results and fill NaN with 0
        all_nodes = pd.DataFrame({'Node': G.nodes})
        time_df_result = pd.merge(all_nodes, df_sorted_diff, how='left', left_on='Node', right_index=True)
        time_df_result['Sorted_Normalized_Difference'].fillna(0, inplace=True)

        # Print the DataFrame
        # print("\nDataFrame for Sorted Normalized Differences:")
        print(time_df_result)
        time_df_result.to_csv(OUT_PUT_FILE_TIMESTAMP, sep='\t', index=False)
        
    else:
        print("\nNo nodes with sufficient data for normalization.")

    #time.visualize_graph(G)

except ValueError as e:
    print(f"Error: {str(e)}")

# Merging DataFrames
final_df = pd.merge(df_ppr, weight_normalized_df, on='Node', how='outer')
final_df = pd.merge(final_df, time_df_result, on='Node', how='outer')
final_df = final_df.rename(columns={'Normalized_Result': 'Weight', 'Sorted_Normalized_Difference': 'Timestamp'})
print(final_df)

########################## ANOMALY SCORE ##################
anomaly_scores, (max_anomaly_node, max_anomaly_score) = ascore.calculate_anomaly_scores(final_df)

# Print the result
#print("Node\tAnomaly Score")
#for node, score in anomaly_scores:
 #   print(f"{node}\t{score}")

#print("\nMaximum Anomaly Score:")
#print("Node:", max_anomaly_node)
#print("Score:", max_anomaly_score)

# Open the output file in write mode and redirect the print statements to the file
with open(OUT_PUT_FILE_ANOMALY_SCORES, 'w') as output_file:
    # Print the result to the output file
    output_file.write("Node\tAnomaly Score\n")
    for node, score in anomaly_scores:
        output_file.write(f"{node}\t{score}\n")

    output_file.write("\nMaximum Anomaly Score:\n")
    output_file.write(f"Node: {max_anomaly_node}\n")
    output_file.write(f"Score: {max_anomaly_score}\n")

# Open the max anomaly file in write mode and save the maximum anomaly score information
with open(OUT_PUT_FILE_MAX_ANOMALY_SCORE, 'w') as max_anomaly_file:
    max_anomaly_file.write(f"Node: {max_anomaly_node}\n")
    max_anomaly_file.write(f"Score: {max_anomaly_score}\n")

# Notify the user that the output has been saved
print(f"Output has been saved to {OUT_PUT_FILE_ANOMALY_SCORES}")
print(f"Maximum anomaly score information has been saved to {OUT_PUT_FILE_MAX_ANOMALY_SCORE}")