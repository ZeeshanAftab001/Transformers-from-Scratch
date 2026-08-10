import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self,model_dim,n_heads,dropout_p):
        super().__init__()

        self.model_dim=model_dim
        self.n_heads=n_heads
        self.dropout_p=dropout_p
        assert model_dim % n_heads == 0 ,"model_dim must be completely divisible by n_heads"
        self.d_k=self.model_dim // n_heads # Dimension of vector seen by each head

        self.w_q=nn.Linear(self.model_dim,self.model_dim)
        self.w_k=nn.Linear(self.model_dim,self.model_dim)
        self.w_v=nn.Linear(self.model_dim,self.model_dim)

        self.w_o=nn.Linear(self.model_dim,self.model_dim)
        self.dropout=nn.Dropout(p=self.dropout_p)

    def attention(self,query,key,value,mask,dropout):

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        value=attention_scores @ value
        return value, attention_scores

    def forward(self,x,mask):

        query=self.w_q(x)
        key=self.w_k(x)
        value=self.w_v(x)

        # (batch_size,seq_len,model_dim) -> (batch_size,seq_len,n_heads,self.d_k) -> (batch_size,n_heads,seq_len,self.d_k) 
        query=query.view(query.shape[0],query.view[1],self.n_heads,self,self.d_k).transpose(1,2)
        key=key.view(query.shape[0],query.view[1],self.n_heads,self,self.d_k).transpose(1,2)
        value=value.view(query.shape[0],query.view[1],self.n_heads,self,self.d_k).transpose(1,2)


        value,self.attention_scores=MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

         # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)