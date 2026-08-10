import torch
import torch.nn as nn
from src.residual_connection import ResidualConnection
from src.layer_norm import LayerNormalization

class EncoderBlock(nn.Module):

    def __init__(self, x, self_attention_block, feed_forward_block, dropout_p):
        super().__init__()

        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.dropout_p=dropout_p

        self.residual_connections=nn.ModuleList([ResidualConnection(x,dropout_p) for _ in range(2)])

    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x : self.self_attention_block(x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)