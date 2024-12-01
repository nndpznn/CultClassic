from data.datasets import prepData, getDataLoaders
from models.neuralNetwork import Model as md
from utils.eval import evaluate
import torch
from torch.optim import Adam
import torch.nn.functional as F

# Grabbing data, prepping with new indexes and creating dataloader from tensors.
userTensor, movieTensor, ratingTensor, nUsers, nMovies = prepData("data/raw/ml-100k/u.data")
dataLoader = getDataLoaders(userTensor, movieTensor, ratingTensor, batchSize=64)

# Initializing our model, with dimensions we can play around with, as well as our SGD function.
model = md(nUsers,nMovies,64,[256,128,64])
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(20):
  for uBatch, mBatch, rBatch in dataLoader:
    predictions = model(uBatch, mBatch)
    rBatch = rBatch.view(-1,1)

    loss = F.mse_loss(predictions, rBatch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
