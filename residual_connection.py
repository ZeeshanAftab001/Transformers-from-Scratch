import torch
import torch.nn as nn
from layer_norm import LayerNorm


class ResidualConnection(nn.Module):

    def __init__(self,dropout_p):
        super().__init__()

        self.dropout=nn.Dropout(p=dropout_p)
        self.norm=LayerNorm()

    def forward(self,x,sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))