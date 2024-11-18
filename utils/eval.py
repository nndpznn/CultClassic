# We'll develop our evaluation functions here.
# We're using mean square error...

from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd

# RMSE: Root Mean Square Error
def rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))

# Precision@k
def precision_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_at_k) & relevant_set) / k

# Recall@k
def recall_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_at_k) & relevant_set) / len(relevant_set)

# NDCG: Normalized Discounted Cumulative Gain
def ndcg(recommended, relevant, k):
    dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(recommended[:k]) if item in relevant])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0

# Evaluate Function
def evaluate(predictions_file, actual_file, k=5):
    predictions = pd.read_csv(predictions_file) 
    actual = pd.read_csv(actual_file) 
    
    # TODO: Change the rating parameter to the actual rating parameter name once model is trained
    pred = predictions['rating'].values 
    act = actual['rating'].values 
    recommended_items = predictions['items']
    relevant_items = actual['items']
    
    print(f"RMSE: {rmse(pred, act):.4f}")
    print(f"Precision@{k}: {precision_at_k(recommended_items, relevant_items, k):.4f}")
    print(f"Recall@{k}: {recall_at_k(recommended_items, relevant_items, k):.4f}")
    print(f"NDCG@{k}: {ndcg(recommended_items, relevant_items, k):.4f}")

# Main Execution
# TODO: Change file paths
if __name__ == "__main__":
    evaluate("predictions.csv", "actual_ratings.csv")