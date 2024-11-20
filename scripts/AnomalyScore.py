import pandas as pd

def calculate_anomaly_scores(data):
    anomaly_scores = []
    max_anomaly_score = float('-inf')
    max_anomaly_node = None

    for i in range(len(data)):
        row = data.iloc[i]
        node = row['Node']
        timestamp_score = row['Timestamp']
        weight_score = row['Weight']
        ppr_value = row['PPR']

        # Avoid division by zero, handle it according to your specific use case
        if timestamp_score * weight_score != 0:
            anomaly_score = ppr_value / (timestamp_score * weight_score)
            if anomaly_score > max_anomaly_score:
                max_anomaly_score = anomaly_score
                max_anomaly_node = node
        else:
            anomaly_score = float('inf')  # Handle division by zero, you can adjust it based on your use case

        anomaly_scores.append((node, anomaly_score))

    return anomaly_scores, (max_anomaly_node, max_anomaly_score)


# Calculate anomaly scores and maximum anomaly score with its corresponding node
#anomaly_scores, (max_anomaly_node, max_anomaly_score) = calculate_anomaly_scores(data)

