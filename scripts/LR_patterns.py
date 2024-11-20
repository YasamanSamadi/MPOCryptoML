import pandas as pd
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Helper function to normalize a list
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

# Function to compute personalized PageRank scores
def compute_ppr(graph):
    return nx.pagerank(graph)

# Function to compute new features F(θ,ω)(vi)
def compute_features(graph, timestamps, weights):
    """
    Computes features including:
    - Fan-in, fan-out, stack, gather-scatter, bipartite patterns
    - Normalized timestamp score (θ)
    - Normalized weight score (ω)
    - Personalized PageRank scores (PPR)
    """
    X = []  # Feature matrix
    y = []  # Labels (for simplicity, we use dummy labels)
    
    # Normalize timestamps and weights
    normalized_timestamps = normalize(timestamps)
    normalized_weights = normalize(weights)

    # Compute Personalized PageRank scores
    ppr_scores = compute_ppr(graph)

    for i, node in enumerate(graph.nodes):
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        neighbors = list(graph.neighbors(node))

        # Fan-in pattern: High in-degree
        is_fan_in = in_degree > out_degree and in_degree > 2

        # Fan-out pattern: High out-degree
        is_fan_out = out_degree > in_degree and out_degree > 2

        # Stack pattern: Alternating in-out connections
        is_stack = len(neighbors) > 2 and all(
            (graph.in_degree(neighbor) > 0 and graph.out_degree(neighbor) > 0) for neighbor in neighbors
        )

        # Gather-scatter: A mix of fan-in and fan-out from the same node
        is_gather_scatter = is_fan_in and is_fan_out

        # Bipartite pattern: Check for connections between two disjoint sets
        is_bipartite = nx.is_bipartite(graph)

        # Compute F(θ,ω)(v_i)
        theta = normalized_timestamps[i]
        omega = normalized_weights[i]
        ppr = ppr_scores.get(node, 0)
        
        features = [
            in_degree,
            out_degree,
            int(is_fan_in),
            int(is_fan_out),
            int(is_stack),
            int(is_gather_scatter),
            int(is_bipartite),
            theta,
            omega,
            ppr,
        ]
        X.append(features)

        # Dummy label: Assume nodes with fan-in or fan-out are anomalous
        label = int(is_fan_in or is_fan_out)
        y.append(label)

    return np.array(X), np.array(y)

# Load dataset from CSV file
csv_file = "your_dataset.csv"  # Replace with your dataset file
data = pd.read_csv(csv_file)

# Ensure the dataset has the required columns
required_columns = ['source', 'target', 'weight', 'timestamp']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the columns: {required_columns}")

# Extract data
sources = data['source']
targets = data['target']
weights = data['weight']
timestamps = data['timestamp']

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph with weights
for source, target, weight in zip(sources, targets, weights):
    G.add_edge(source, target, weight=weight)

# Compute features and labels
X, y = compute_features(G, timestamps, weights)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Analyze patterns for all nodes
for node, features in zip(G.nodes, X):
    prediction = model.predict([features])
    print(f"Node {node}: Features {features}, Predicted Pattern {prediction[0]}")
