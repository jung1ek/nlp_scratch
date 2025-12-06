import torch
import torch.nn as nn
import copy
import math

def clones(x,N):
    return nn.ModuleList([copy.deepcopy(x) for _ in range(N)])

def attention(query,key,value,mask=None,dropout=None):
    dk = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    attn_weights = scores.softmax(dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights,value), attn_weights

def subsequent_mask(size):
    shape = (1,size,size)
    subsequent_mask = torch.triu(torch.ones(shape),diagonal=1).type(
        torch.uint8)
    return subsequent_mask == 0

class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,heads,dropout_p=0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model//heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self,query,key,value,mask):
        batch_size = query.size(0)
        # do all linear proj in batch from d_model => h * d_k
        query, key, value = [
            lin(x).view(batch_size, -1, self.heads,self.d_k).transpose(1,2)
            for lin, x in zip(self.linears, (query,key,value))
        ]
        x, attn_weights = attention(query,key,value,mask=mask)
        output = x.transpose(1,2).contiguous().view(batch_size,-1,self.d_k*self.heads)
        return self.linears[-1](output)

class SublayerConnection(nn.Module):

    def __init__(self,dropout_p=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = LayerNorm()
    
    def forward(self,x,sublayer):
        # eg callable sublayer; ffn and mha
        return x + self.dropout(sublayer(self.layer_norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std+self.eps) + self.b_2
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class PositionalEncoding(nn.Module):

    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0,d_model,2) * -(math.log(10000.0)/d_model)
        )
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:,: x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

class Embeddings(nn.Module):

    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model

    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)