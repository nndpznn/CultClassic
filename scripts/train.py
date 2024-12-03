from data.datasets import prepData, getDataLoaders
from models.neuralNetwork import Model as md
from utils.eval import evaluate
import torch
from torch.optim import Adam
import torch.nn.functional as F
import os

# Create the directory if it doesn't exist
save_dir = "./scripts/savedWeights/"
os.makedirs(save_dir, exist_ok=True)

# Grabbing data, prepping with new indexes and creating dataloader from tensors.
userTensor, movieTensor, ratingTensor, nUsers, nMovies = prepData("data/raw/ml-100k/u.data")
dataLoader = getDataLoaders(userTensor, movieTensor, ratingTensor, batchSize=64)

# Initializing our model, with dimensions we can play around with, as well as our SGD function.
model = md(nUsers,nMovies,64,[256,128,64])
optimizer = Adam(model.parameters(), lr=0.001)

try:
    model.load_state_dict(torch.load(os.path.join(save_dir, "final_model.pth")))
    print("Loaded model weights from previous training.")
except FileNotFoundError:
    print("No saved weights found. Starting fresh.")

model.eval() 

for epoch in range(20):
  for uBatch, mBatch, rBatch in dataLoader:
    predictions = model(uBatch, mBatch)
    rBatch = rBatch.view(-1,1)

    loss = F.mse_loss(predictions, rBatch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

      # Saving the model checkpoint every 5 epochs
  if (epoch + 1) % 5 == 0:
      checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
      torch.save(model.state_dict(), checkpoint_path)
      print(f"Model saved to {checkpoint_path}")

# Save the final model
final_model_path = os.path.join(save_dir, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")