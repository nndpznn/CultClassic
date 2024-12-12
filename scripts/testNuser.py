import sys
import torch
from data.datasets import prepData
from utils.recommend import recommend_top_k_movies
from models.neuralNetwork import Model

def moviesForNUser(userID, k):
	userTensor, movieTensor, ratingTensor, nUsers, nMovies, idToTitle = prepData("data/raw/ml-100k/u.data", "data/raw/ml-100k/u.item")
	model = Model(nUsers, nMovies, embeddingDim=64, hiddenDims=[256,128,64])
      
	# Load trained weights (ensure the path is correct)
	model.load_state_dict(torch.load("./scripts/savedWeights/final_model.pth"))
	model.eval()  # Set model to evaluation mode

	top_k_movies = recommend_top_k_movies(userID, model, nMovies, k)

	print(f"Top {k} movies for user {userID}:")
	for index, movieID in enumerate(top_k_movies):
		movie_title = idToTitle.get(movieID, "Unknown Movie")
		print(f"{index + 1}. {movie_title}")


if __name__ == "__main__":
    # Ensure arguments are passed correctly
    if len(sys.argv) < 3:
        print("Usage: python script.py <userID> <k>")
        sys.exit(1)

    # Parse arguments
    userID = int(sys.argv[1])
    k = int(sys.argv[2])

    # Call the function
    moviesForNUser(userID, k)
