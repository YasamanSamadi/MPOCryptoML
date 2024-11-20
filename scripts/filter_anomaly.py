import networkx as nx
import pandas as pd

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

def main():
    # Load data from anomaly_score.csv
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

    # Save the results to new CSV files
    filtered_nodes_path = "filtered_laundering_accounts.csv"
    anomaly_scores_path = "anomaly_scores_with_filtering.csv"

    pd.DataFrame({'Filtered_Nodes': filtered_nodes}).to_csv(filtered_nodes_path, index=False)
    anomaly_scores_df.to_csv(anomaly_scores_path, index=False)

    # Display results
    print(f"Filtered Laundering Accounts (Nodes) saved to {filtered_nodes_path}.")
    print(f"Anomaly Scores saved to {anomaly_scores_path}.")

if __name__ == "__main__":
    main()
