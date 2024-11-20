import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import networkx as nx #NEW
import ppr as ppr #NEW 1


#NEW box1
def initialize_values_for_multiple_sources(graph, sources):
    # Existing initialization code
    probability_per_source = 1 / len(sources)
    residue_values = {(source, u): probability_per_source if (source, u) != (source, source) else 1 for source in sources for u in graph.nodes}
    reserve_values = {(source, u): 0 for source in sources for u in graph.nodes}
    PPR_values = {(source, v): 0 for source in sources for v in graph.nodes}
    random_walks_numbers = {(source, u): 0 for source in sources for u in graph.nodes}
    
    # Calculate PageRank scores
    pagerank_scores = nx.pagerank(graph)
    
    # Integrate PageRank scores into PPR_values
    for source in sources:
        for v in graph.nodes:
            PPR_values[(source, v)] = pagerank_scores[v]

    return residue_values, reserve_values, PPR_values, random_walks_numbers


#END NEW box1

def find_anomalous_nodes(data):
    # Drop any rows with NaN values
    data = data.dropna()

    # Standardize the data
    features = data[['PPR', 'Weight', 'Timestamp']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Fit Elliptic Envelope model
    model = EllipticEnvelope(contamination=0.1)  # Adjust contamination as needed
    model.fit(reduced_features)

    # Predict anomalies
    anomalies = model.predict(reduced_features)

    # Extract anomalous nodes
    anomaly_nodes = data[anomalies == -1]['Node']

    return anomaly_nodes

# Example usage:
#anomaly_nodes = find_anomalous_nodes(final_df)
#print("Anomalous Nodes:")
#print(anomaly_nodes)
#anomaly_nodes.to_csv('anomalous_nodes.txt', index=False, header=True)
