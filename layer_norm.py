import torch 
import torch.nn as nn

class LayerNormalization(nn.Module):
    '''
        Input : 
            model_din : int -> dimentions of the vector e.g 512
                eps : float -> epsilon inorder to normalize values for stability
        Output : 
            a normalized embedding vector of shape same as input
    '''
    def __init__(self,model_dim,eps = 1e-5):

        super().__init__()
        self.eps=eps

        self.alpha = nn.Parameter(torch.ones(model_dim)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(model_dim)) # Added

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * out + self.beta