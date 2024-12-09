# We'll develop our evaluation functions here.
# We're using mean square error...

from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd

#MAE = Mean Absolute Error
def mae(predictions, actual):
    return np.mean(np.abs(predictions- actual))

# RMSE: Root Mean Square Error
def rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))

# Precision@k
def precision_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = relevant

    positive_mask = [rec >= 4.0 for rec in recommended_at_k]

    truePos = 0
    for ratingIndex in range(len(recommended_at_k)):
        if recommended_at_k[ratingIndex] >= 4.0 and relevant_set[ratingIndex] >= 4.0:
                truePos += 1
    
    return truePos / k

# Recall@k
def recall_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = relevant

    truePos_falseNeg = [rec >= 4.0 for rec in relevant_set]

    truePos = 0
    for ratingIndex in range(len(recommended_at_k)):
        if recommended_at_k[ratingIndex] >= 4.0 and relevant_set[ratingIndex] >= 4.0:
                truePos += 1

    return truePos / len(truePos_falseNeg)

# NDCG: Normalized Discounted Cumulative Gain
def ndcg(recommended, relevant, k):
    dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(recommended[:k]) if item in relevant])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant), k))])
    return dcg / idcg if idcg > 0 else 0

# NOLAN'S NDCG: Normalized Discounted Cumulative Gain
# def ndcg(recommended, relevant, k):
#     # Sort recommended items by predicted rating (descending order)
#     # recommended could be a list of predicted ratings
#     # Get the indices that would sort the recommended items in descending order
#     sorted_indices = np.argsort(recommended)[::-1]

#     # DCG: Discounted Cumulative Gain at position k
#     dcg = 0
#     for idx in range(k):
#         item = sorted_indices[idx]  # Get the index of the item
#         relevance = relevant[item]  # Use the actual relevance score from the 'relevant' array
#         dcg += relevance / np.log2(idx + 2)  # +2 for 1-based log2

#     # IDCG: Ideal Discounted Cumulative Gain at position k
#     sorted_relevant = np.sort(relevant)[::-1]  # Sort relevance in descending order
#     idcg = 0
#     for idx in range(k):
#         relevance = sorted_relevant[idx]  # Use the sorted relevance for the ideal ranking
#         idcg += relevance / np.log2(idx + 2)  # +2 for 1-based log2

#     # Return normalized DCG
#     return dcg / idcg if idcg > 0 else 0

# Evaluate Function
def evaluate(predictions, actual, k):
    # Handle numpy array inputs
    if isinstance(predictions, np.ndarray) and isinstance(actual, np.ndarray):
        correct_indices = predictions.argsort()[::-1][:k]  # Top-k indices
        recommended_items = predictions[correct_indices]  # Sorted Predictions
        relevant_items = actual[correct_indices]           # Sorted Actuals
        
    # elif isinstance(predictions, pd.DataFrame) and isinstance(actual, pd.DataFrame):
    #     pred = predictions['rating'].values
    #     act = actual['rating'].values
    #     recommended_items = predictions['items']
    #     relevant_items = actual['items']
    else:
        raise ValueError("Unsupported input types. Use numpy arrays or pandas DataFrames.")

    print(f'MAE: {mae(predictions, actual):.4f}')
    print(f"RMSE: {rmse(predictions, actual):.4f}")
    print(f"Precision@{k}: {precision_at_k(recommended_items, relevant_items, k):.4f}")
    print(f"Recall@{k}: {recall_at_k(recommended_items, relevant_items, k):.4f}")
    print(f"NDCG@{k}: {ndcg(recommended_items, relevant_items, k):.4f}")
    # print(f"Recommended items: {recommended_items[:10]}")
    # print(f"Relevant items: {relevant_items[:10]}")


# Main Execution
# TODO: Change file paths
if __name__ == "__main__":
    evaluate("predictions.csv", "actual_ratings.csv")