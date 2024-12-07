import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from data.datasets import prepData
from models.neuralNetwork import Model  # Assuming this is your neural network model
from utils.eval import rmse, precision_at_k, recall_at_k, ndcg, evaluate

# 1. Load the trained model
userTensor, movieTensor, ratingTensor, nUsers, nMovies = prepData("data/raw/ml-100k/u.data")
model = Model(nUsers, nMovies, embeddingDim=64, hiddenDims=[256,128,64])  # Adjust based on your model
model.load_state_dict(torch.load('scripts/savedWeights/final_model.pth', weights_only=True)) # Load the trained model
model.eval()  # Set the model to evaluation mode

# 2. Load the test data
test_data = pd.read_csv('data/raw/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

# 3. Preprocess the test data
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

# Fit encoders
test_data['userId'] = user_encoder.fit_transform(test_data['userId'])
test_data['movieId'] = movie_encoder.fit_transform(test_data['movieId'])
# Ensure 'rating' column is numeric, forcing errors to NaN and then handling them
test_data['rating'] = pd.to_numeric(test_data['rating'], errors='coerce')

# After converting, you can drop rows with NaN values (if any) or handle them in another way
test_data = test_data.dropna(subset=['rating'])



# Convert to tensors
user_tensor = torch.tensor(test_data['userId'].values, dtype=torch.long)
movie_tensor = torch.tensor(test_data['movieId'].values, dtype=torch.long)
# Now convert the 'rating' column to a tensor
rating_tensor = torch.tensor(test_data['rating'].values, dtype=torch.float)

# 4. Create a TensorDataset and DataLoader for batching
test_dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # DataLoader for batching

# TODO 3. Call in evaluation function from eval.py and return the actuals and predictions rating
def evaluate_model(model, test_loader):
    actuals, predictions = [], []
    with torch.no_grad():  # Disable gradient tracking during inference
        for userIds, movieIds, ratings in test_loader:
            preds = model(userIds, movieIds).squeeze()  # Get model predictions
            actuals.extend(ratings.numpy())  # Collect true ratings
            predictions.extend(preds.numpy())  # Collect predicted ratings

    actuals, predictions = np.array(actuals), np.array(predictions)
    return actuals, predictions


# 4. Calculate MAE and RMSE metrics
def calculate_metrics(actuals, predictions):
    mae = np.mean(np.abs(predictions- actuals))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))  # Root Mean Squared Error
    return mae, rmse

# 5. Run the evaluation
actuals, predictions = evaluate_model(model, test_loader)  # Get the predictions and actual ratings

# Calculate the metrics
mae, rmse = calculate_metrics(actuals, predictions)

# 6. Output the results
print(f'MAE: {mae:.4f}')
evaluate(predictions=predictions, actual=actuals, k=1000)
print(f"Number of test samples: {len(actuals)}")
print(f"Example Actuals: {actuals[:5]}")
print(f"Example Predictions: {predictions[:5]}")