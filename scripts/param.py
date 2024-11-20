import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def calculate_anomaly_scores_with_filtering(df):
    """
    Calculates anomaly scores for nodes and filters laundering accounts.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Node', 'PPR', 'Weight', 'Timestamp'].

    Returns:
        filtered_nodes (list): Nodes flagged as laundering accounts.
        df_anomaly_scores (pd.DataFrame): DataFrame with calculated anomaly scores.
    """
    # Compute F(θ, ω)(vi) as the product of Weight and Timestamp
    df['F(θ,ω)'] = df['Weight'] * df['Timestamp']
    
    # Avoid division by zero
    df['F(θ,ω)'] = df['F(θ,ω)'].replace(0, 1e-9)
    
    # Calculate anomaly score σ(vi) = π(vi) / F(θ,ω)(vi)
    df['Anomaly_Score'] = df['PPR'] / df['F(θ,ω)']
    
    # Determine a threshold (e.g., top 1% of scores)
    threshold = df['Anomaly_Score'].quantile(0.99)
    
    # Filter nodes exceeding the threshold
    filtered_nodes = df[df['Anomaly_Score'] > threshold]['Node'].tolist()
    
    return filtered_nodes, df[['Node', 'Anomaly_Score']]

def load_labels(labels_path):
    """
    Load the true labels from the dataset.csv file.
    
    Parameters:
        labels_path (str): The path to the labels dataset file.
    
    Returns:
        pd.DataFrame: DataFrame containing the wallet address and the true label.
    """
    try:
        labels_df = pd.read_csv(labels_path)
        print(f"Labels loaded successfully from {labels_path}.")
        return labels_df
    except FileNotFoundError:
        print(f"Error: File {labels_path} not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def evaluate_performance(true_labels, predicted_labels):
    """
    Evaluates the classification performance using TP, FP, TN, FN and metrics like precision, recall, etc.
    
    Parameters:
        true_labels (list): The true labels of the nodes.
        predicted_labels (list): The predicted labels (malicious or not) of the nodes.
    
    Returns:
        dict: A dictionary with all evaluation metrics.
    """
    # Confusion Matrix: TP, FP, TN, FN
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    # Calculate other metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # ROC-AUC score calculation (requires binary labels for classification)
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

def main():
    # Load anomaly score data
    try:
        data_path = "anomaly_score.csv"  # Replace with the actual file path if needed
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully from {data_path}.")
    except FileNotFoundError:
        print(f"Error: File {data_path} not found. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Check for required columns in the CSV
    required_columns = {'Node', 'PPR', 'Weight', 'Timestamp'}
    if not required_columns.issubset(df.columns):
        print(f"Error: The file must contain the following columns: {required_columns}")
        return

    # Calculate anomaly scores and filter laundering accounts
    filtered_nodes, anomaly_scores_df = calculate_anomaly_scores_with_filtering(df)

    # Load the true labels from dataset.csv (replace with your actual labels file)
    labels_path = "dataset.csv"  # Replace with your labels file
    labels_df = load_labels(labels_path)
    
    if labels_df is None:
        return

    # Merge the true labels with the detected malicious nodes
    # Assuming the labels dataset has columns 'Node' and 'Label' where 1 = malicious, 0 = non-malicious
    merged_df = pd.merge(labels_df, pd.DataFrame({'Node': filtered_nodes, 'Predicted_Label': [1] * len(filtered_nodes)}), on='Node', how='left')
    merged_df['Predicted_Label'] = merged_df['Predicted_Label'].fillna(0).astype(int)
    
    # Get the true labels (0 = non-malicious, 1 = malicious)
    true_labels = merged_df['Label'].tolist()
    predicted_labels = merged_df['Predicted_Label'].tolist()

    # Evaluate the performance
    metrics = evaluate_performance(true_labels, predicted_labels)

    # Display the results
    print(f"Evaluation Metrics: {metrics}")
    
    # Save the results to CSV
    performance_path = "evaluation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(performance_path, index=False)
    print(f"Performance evaluation saved to {performance_path}.")

if __name__ == "__main__":
    main()
