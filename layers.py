import torch 
import torch.nn as nn

class LayerNorm(nn.Module):
    
    def __init__(self,model_dim,eps = 1e-5):

        super().__init__()
        self.eps=eps

        self.alpha = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * out + self.beta