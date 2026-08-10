import torch 
import torch.nn as nn


class PositionalEncoder(nn.Module):
    '''
        Input : 
            model_din : int -> dimentions of the vector e.g 512
            seq_len : int -> total number of embedding vectors in the sequence
        Output : 
            an embedding vector of 512 dimentions + with postional information
    '''
    
    def __init__(self,model_dim,seq_len,dropout_p): 

        super().__init__()
        self.model_dim=model_dim
        self.seq_len=seq_len
        self.dropout_p=dropout_p
        self.dropout=nn.Dropout(p=self.dropout_p)

        pos_emb=torch.zeros(seq_len,model_dim) # (sequence_length,512)

        # postion for every single embedding to add postion encoding
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # tensor(seq_len,1)
        div_term=torch.exp(torch.arange(0,self.model_dim,2).float() * -torch.log(torch.tensor(10000)) /model_dim)

        # apply sin to every even postion
        pos_emb[:,0::2]=torch.sin(position * div_term)
        # apply cos to every odd position
        pos_emb[:,1::2]=torch.cos(position * div_term)

        pos_emb = pos_emb.unsqueeze(0) # adding dimention for batch

        self.register_buffer("pos_emb",pos_emb)

    def forward(self,x):
        out=x + (self.pos_emb[:,:x.shape[1],:])
        return self.dropout(out)
    
if __name__=="__main__":

    x=torch.randn(1,5,512)

    p=PositionalEncoder(
        model_dim=512,
        seq_len=100,
        dropout_p=0.1
    )

    print(p(x).shape)
    