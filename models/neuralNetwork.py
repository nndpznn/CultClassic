import torch
import torch.nn as nn
# import torch.nn.functional as F

class Model(nn.Module):
    """
    ARGS:
        - nUsers and nItems, the number of users and items we're inputting.
        - embeddingDim, the dimensions for both our user and item embeddings.
        - hiddenDims, a list of dimensions for each hidden layer, respectively.
            - hiddenDims is parameterized so that we can easily change how many layers
            we have and how many nodes are in each layer.
    """
    def __init__(self, nUsers, nItems, embeddingDim, hiddenDims):
        super(Model, self).__init__()
        
        # Here, the models attributes are two "embeddings". These are basically 
        # dictionaries containing the vectors we'll be making out of our movies and users.
        self.userEmbedding = nn.Embedding(nUsers,embeddingDim)
        self.itemEmbedding = nn.Embedding(nItems,embeddingDim)

        # Defining our hidden layers...
        layers = []

        # We're using the concatenation technique, adding the user and item embeddings
        # together so that their relationships can be discovered by the model.
        inputDim = embeddingDim * 2

        # For each layer specified, we're adding a new fully-connected layer of size specified.
        for hiddenDim in hiddenDims:
            layers.append(nn.Linear(inputDim, hiddenDim))
            layers.append(nn.ReLU())
            inputDim = hiddenDim

        layers.append(nn.Linear(inputDim, 1))

        self.network = nn.Sequential(*layers)


    def forward(self, userIDs, itemIDs):
        
        userVecs = self.userEmbedding(userIDs)
        itemVecs = self.itemEmbedding(itemIDs)

        # Concatenating the layers here.
        catted = torch.cat([userVecs,itemVecs], dim=-1)

        return self.network(catted)
    

        
    