import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    '''
        Input : 
            model_din : int -> dimentions of the vector e.g 512
            dropout_p : float -> dropout percentage
            ff_dim : int -> dimention of hidden layer
        Output : 
            a vector of shape same as input
        '''
    def __init__(self,model_dim,dropout_p=0.5,ff_dim=2048):
        super().__init__()
        self.model_dim=model_dim
        self.dropout_p=dropout_p

        self.network=nn.Sequential(
            nn.Linear(model_dim,ff_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(ff_dim),
        )

    def forward(self,x):
        return self.network(x)