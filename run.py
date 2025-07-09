from models import Encoder
import torch

enc = Encoder(4,5)

h_enc = enc(torch.rand(3,4)) # shape (hj) j=3, n= id*2 (8)
n = 5 # output's state dimension
w = torch.rand(n,n)
u = torch.rand(n,2*n)
v = torch.rand(n)
s = torch.zeros(n) # initial state of decoder layer
e = torch.tanh(w@s+h_enc@u.T) @ v
a = torch.exp(e)/torch.sum(torch.exp(e),dim=-1)
c = a@h_enc
print(c.shape)
