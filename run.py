from models import Encoder
import torch

enc = Encoder(4,5)

print(enc(torch.rand(3,4)))

print(reversed(torch.rand(4)))