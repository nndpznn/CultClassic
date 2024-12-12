import pandas as pd
from sklearn.model_selection import train_test_split as tts
import torch
from torch.utils.data import DataLoader, TensorDataset

def prepData(rating_file_path, movie_file_path, delimiter="\t"):

	# "raw/ml-100k/u.data"
	data = pd.read_csv(rating_file_path, sep=delimiter)
	# print(data.head(10))

	movies = pd.read_csv(movie_file_path, sep="|", encoding="latin-1", header=None, names=[
	"movieID", "title", "release_date", "video_release_date", "IMDb_URL",
	"unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
	"Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
	"Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ])

	# Would re-map user IDs to new IDs, but we know all IDs are present.
	remapUsers = {u: i for i,u in enumerate(data["userID"].unique())}
	remapMovies = {u: i for i,u in enumerate(data["movieID"].unique())}
	data["userID"] = data["userID"].map(remapUsers)
	data["movieID"] = data["movieID"].map(remapMovies)

	# Casting data to correct types for use 
	data["userID"] = data["userID"].astype(int)
	data["movieID"] = data["movieID"].astype(int)
	data["rating"] = data["rating"].astype(float)

	movies["new_movieID"] = movies["movieID"].map(remapMovies)
	id_to_title = movies.set_index("new_movieID")["title"].to_dict()

	# Using train_test_split() to randomly partition 1/5 of the data for testing.
	trainingData, testingData = tts(data, test_size=0.2, random_state=None)

	trainTensorU = torch.tensor(trainingData["userID"].values, dtype=torch.long)
	trainTensorM = torch.tensor(trainingData["movieID"].values, dtype=torch.long)
	trainTensorR = torch.tensor(trainingData["rating"].values, dtype=torch.float)

	# testTensorU = torch.tensor(trainingData["userID"].values, dtype=torch.long)
	# testTensorM = torch.tensor(trainingData["movieID"].values, dtype=torch.long)
	# testTensorR = torch.tensor(trainingData["rating"].values, dtype=torch.float)

	return trainTensorU, trainTensorM, trainTensorR, len(data["userID"].unique()), len(data["movieID"].unique()), id_to_title

def getDataLoaders(tensorU, tensorM, tensorR, batchSize=64):
	trainDataset = TensorDataset(tensorU, tensorM, tensorR)
	return DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

