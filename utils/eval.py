# We'll develop our evaluation functions here.
# We're using mean square error...
import torch
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score, recall_score
import numpy as np
import pandas as pd
from utils.recommend import recommend_top_k_movies
import random

#MAE = Mean Absolute Error
def mae(predictions, actual):
    return np.mean(np.abs(predictions - actual))

# RMSE: Root Mean Square Error
def rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))

# Precision@k
def precision_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_at_k) & relevant_set) / len(recommended_at_k)

# Recall@k
def recall_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_at_k) & relevant_set) / len(relevant_set)

def ndcg(recommended, relevant, k):
    dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(recommended[:k]) if item in relevant])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant), k))])  # Ideal DCG

    # Handle edge cases where idcg is 0 (to avoid division by zero)
    return dcg / idcg if idcg > 0 else 0

# Evaluate Function
def evaluate(model, df, k, nMovies):
    # Generate recommendations for each user using recommend_top_k_movies
    df['user_id'] = df['user_id'].astype(int)
    df['movie_id'] = df['movie_id'].astype(int)

    user_ids = df['user_id'].values
    movie_ids = df['movie_id'].values
    actual_ratings = df['rating'].values

    recommended_items = []
    relevant_items = []

    for user_id in user_ids:
        top_k_movies = recommend_top_k_movies(user_id, model, nMovies, k)
        recommended_items.append(top_k_movies)
        
        # Extract relevant items for this user
        relevant_items_for_user = df[df['user_id'] == user_id]['movie_id'].values
        relevant_items.append(relevant_items_for_user)
    
    # Flatten the list of recommended and relevant items for the evaluation metrics
    recommended_items_flat = [item for sublist in recommended_items for item in sublist]
    relevant_items_flat = [item for sublist in relevant_items for item in sublist]
    
    # Now evaluate using Precision@k, Recall@k, and NDCG
    print(f"Precision@{k}: {precision_at_k(recommended_items_flat, relevant_items_flat, k):.4f}")
    # print(f"SciKit Precision@{k}: {precision_score(relevant_items_flat, recommended_items_flat):.4f}")
    print(f"Recall@{k}: {recall_at_k(recommended_items_flat, relevant_items_flat, k):.4f}")
    # print(f"SciKit Recall@{k}: {recall_score(relevant_items_flat, recommended_items_flat):.4f}")
    print(f"NDCG@{k}: {ndcg(recommended_items_flat, relevant_items_flat, k):.4f}")


    # Sample a fixed number of user-movie pairs
    sample_size = 500  # Adjust based on memory limits
    sampled_indices = random.sample(range(len(actual_ratings)), sample_size)

    sampled_user_ids = user_ids[sampled_indices]
    sampled_movie_ids = movie_ids[sampled_indices]
    sampled_actual_ratings = actual_ratings[sampled_indices]

    # Compute predictions for the sampled pairs
    user_tensor = torch.tensor(sampled_user_ids, dtype=torch.long)
    movie_tensor = torch.tensor(sampled_movie_ids, dtype=torch.long)
    predictions = model(user_tensor, movie_tensor).detach().numpy()
    
    # Compute and print MAE and RMSE
    print(f"Sampled MAE: {mae(predictions, sampled_actual_ratings):.4f}")
    print(f"Sampled RMSE: {rmse(predictions, sampled_actual_ratings):.4f}")


# Main Execution
# TODO: Change file paths
if __name__ == "__main__":
    evaluate("predictions.csv", "actual_ratings.csv")