import pandas as pd
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv("raw/ml-100k/u.data", sep="\t")
# print(data.head(10))

# Casting data to correct types for use 
data["userID"] = data["userID"].astype(int)
data["movieID"] = data["movieID"].astype(int)
data["rating"] = data["rating"].astype(float)

# data.head(), data.info()

# Using train_test_split() to randomly partition 1/5 of the data for testing.
trainingData, testingData = tts(data, test_size=0.2, random_state=None)



