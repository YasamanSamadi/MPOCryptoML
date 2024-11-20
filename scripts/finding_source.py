import pandas as pd
import networkx as nx
import ast

# Read the dataset into a DataFrame
df = pd.read_csv('Eth_dataset_G.csv')

# Function to extract relevant information from the 'Attributes' column
def is_source(row):
    attributes = ast.literal_eval(row)
    sent_txns = attributes.get('sent_txns', 0)
    return sent_txns > 0

# Apply the function to create a new column indicating if the node is a source
df['Is_Source'] = df['Attributes'].map(is_source)

# Filter the DataFrame to get only the rows where the node is a source
source_df = df[df['Is_Source']]

# Create a directed graph from the filtered DataFrame
G = nx.from_pandas_edgelist(source_df, source='Source', target='Target', create_using=nx.DiGraph())

# Write the source nodes to a CSV file
with open('sources.csv', 'w') as f:
    for node in G.nodes():
        f.write(str(node) + '\n')

print("Source Nodes written to sources.csv.")
