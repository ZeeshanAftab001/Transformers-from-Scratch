import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    '''
        Input : 
            model_din : int -> dimentions of the vector e.g 512
            vocal_size : int -> total number of unique words in the data
        Output : 
            an embedding vector of 512 dimentions
    '''
    def __init__(self,model_dim,vocab_size): 

        super().__init__()
        self.model_dim=model_dim
        self.vocab_size=vocab_size
        self.embeddings=nn.Embedding(vocab_size,model_dim)

    def forward(self,X):

        return self.embeddings(X) * torch.sqrt(self.model_dim) # something done in the paper 


