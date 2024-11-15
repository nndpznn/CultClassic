import pandas as pd

movieData = pd.read_csv("raw/ml-100k/u.data", sep="\t")
print(movieData.head(10))