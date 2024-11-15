import pandas as pd

movieData = pd.read_csv("raw/ratings.csv")
print(movieData.head(10))