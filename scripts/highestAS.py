import pandas as pd
# Read the data from the anomalyscore.txt file
with open('0.1_anomalyscore.txt', 'r') as file:
    lines = file.readlines()

# Remove the header line
lines = lines[1:]

# Parse the data into a dictionary with node IDs as keys and anomaly scores as values
data = {}
for line in lines:
    # Split the line by tab ("\t") and check if it contains at least two values
    values = line.strip().split('\t')
    if len(values) >= 2:
        node, score = values
        # Convert node ID to integer by removing ".0" if present
        node_id = int(float(node))
        # Check if anomaly score is not "inf"
        if score != 'inf':
            data[node_id] = float(score)

# Sort the data based on anomaly scores in descending order
sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

# Extract the node IDs of the highest 100 anomaly scores
top_n = 247
highest_nodes = [node for node, _ in sorted_data[:top_n]]

# Write the node IDs to the "highestAS.txt" file
with open('0.1_highestAS.txt', 'w') as file:
    for node in highest_nodes:
        file.write(f"{node}\n")
#old but gold 


