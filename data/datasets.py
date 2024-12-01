import pandas as pd
from sklearn.model_selection import train_test_split as tts
import torch
from torch.utils.data import DataLoader, TensorDataset

def prepData(file_path, delimiter="\t"):

	# "raw/ml-100k/u.data"
	data = pd.read_csv(file_path, sep=delimiter)
	# print(data.head(10))

	# Would re-map user IDs to new IDs, but we know all IDs are present.
	remapUsers = {u: i for i,u in enumerate(data["userID"].unique())}
	remapMovies = {u: i for i,u in enumerate(data["movieID"].unique())}
	data["userID"] = data["userID"].map(remapUsers)
	data["movieID"] = data["movieID"].map(remapMovies)

	# Casting data to correct types for use 
	data["userID"] = data["userID"].astype(int)
	data["movieID"] = data["movieID"].astype(int)
	data["rating"] = data["rating"].astype(float)

	# data.head(), data.info()

	# Using train_test_split() to randomly partition 1/5 of the data for testing.
	trainingData, testingData = tts(data, test_size=0.2, random_state=None)

	trainTensorU = torch.tensor(trainingData["userID"].values, dtype=torch.long)
	trainTensorM = torch.tensor(trainingData["movieID"].values, dtype=torch.long)
	trainTensorR = torch.tensor(trainingData["rating"].values, dtype=torch.float)

	# testTensorU = torch.tensor(trainingData["userID"].values, dtype=torch.long)
	# testTensorM = torch.tensor(trainingData["movieID"].values, dtype=torch.long)
	# testTensorR = torch.tensor(trainingData["rating"].values, dtype=torch.float)

	return trainTensorU, trainTensorM, trainTensorR, len(data["userID"].unique()), len(data["movieID"].unique())

def getDataLoaders(tensorU, tensorM, tensorR, batchSize=64):
	trainDataset = TensorDataset(tensorU, tensorM, tensorR)
	return DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

