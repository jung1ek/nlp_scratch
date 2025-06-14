import torch
from torch import nn

class RNN(nn.Module):

    def __init__(self, embed_dim,hidden_dim):
        super().__init__()
        self.W_h = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.W_x = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.W_y = nn.Parameter(torch.rand(embed_dim,hidden_dim))
        self.B_h = nn.Parameter(torch.rand(hidden_dim)) 
        self.B_y = nn.Parameter(torch.rand(embed_dim))
        self.hidden_dim=hidden_dim
    
    def forward(self,x):
        h = torch.zeros((self.hidden_dim))
        output = []
        for i in range(x.shape[0]):
            h = torch.tanh((self.W_h@h)+(self.W_x@x[i])+self.B_h)
            output.append(torch.softmax((self.W_y@h)+self.B_y))
        return h,torch.stack(output)
    
class LSTM(nn.Module):

    def __init__(self,embed_dim,hidden_dim):
        super().__init__()
        # forgot gate, weights for what to forgot from long memory
        self.Uf = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.Wf = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.Bf = nn.Parameter(torch.rand(hidden_dim))

        # input gate, new info add to long memory
        self.Ug = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.Wg = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.Bg = nn.Parameter(torch.rand(hidden_dim))

        # output gate, how much info to use from long memory to hidden state
        self.Uq = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.Wq = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.Bq = nn.Parameter(torch.rand(hidden_dim))

        # weights for state gate, how much long term memory retain new info
        self.U = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.W = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.B = nn.Parameter(torch.rand(hidden_dim))

        self.hidden_dim = hidden_dim

    def forward(self,x):
        h = torch.zeros(self.hidden_dim) # init hidden state(short memory)
        s = torch.zeros(self.hidden_dim) # init cell state (long memory)
        ouput = []
        for i in range(x.shape[0]):
            # how much to forgot from prev long memory
            f = torch.sigmoid((self.Wf@h)+(self.Uf@x[i])+self.Bf)

            # how much info to include to long memory , from input.
            c = torch.sigmoid((self.W@h)+(self.U@x[i])+self.B)

            # new info from new input to include in long memory
            g = torch.sigmoid((self.Wg@h)+(self.Ug@x[i])+self.Bg)

            # new long term memory
            s = f*s+g*c

            # output gate, decides how much of the cell state should influence the short-term memory
            q = torch.sigmoid((self.Wq@h)+(self.Uq@x[i])+self.Bq)

            # new short term memory 
            h = torch.tanh(s)*q

            ouput.append(h)
        return torch.stack(ouput)
    

class GRU(nn.Module):

    def __init__(self,embed_dim,hidden_dim):
        super().__init__()
        # update 
        # reset

        # update gate weights,
        self.Uu = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.Wu = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.Bu = nn.Parameter(torch.rand(hidden_dim))

        # reset gate weights 
        self.Ur = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.Wr = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.Br = nn.Parameter(torch.rand(hidden_dim))

        # candidate state (proposed new hidden state)
        self.U = nn.Parameter(torch.rand(hidden_dim,embed_dim))
        self.W = nn.Parameter(torch.rand(hidden_dim,hidden_dim))
        self.B = nn.Parameter(torch.rand(hidden_dim))
        
        self.hidden_dim = hidden_dim
    def forward(self,x):
        h= torch.zeros(self.hidden_dim)
        output = []
        for i in range(x.shape[0]):
            # how much new state to keep vs how much of previous state to retain
            u = torch.sigmoid(self.Uu@x[i]+self.Wu@h+self.Bu)

            # how to of previous state info to forgot for candidate calculation
            r = torch.sigmoid(self.Ur@x[i]+self.Wr@h+self.Br)

            # new sate
            # reset gate is applied to prev hideen state to forgot past info.
            candidate = torch.sigmoid(self.U@x[i]+r*(self.W@h)+self.B)

            # calc new hidden state,
            # If u ≈ 1 → keep mostly old hidden state; if u ≈ 0 → replace with new candidate
            h = u*h+(1-u)*candidate
            output.append(h)

        return h,torch.stack(output)

class RNNv2(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers=1):
        super().__init__()
        # Initialize weights for hidden-to-hidden transitions for each layer
        self.W_h = nn.Parameter(torch.randn(num_layers,hidden_size,hidden_size))
        # Initialize weights for input-to-hidden transition for the first layer
        self.W_x = nn.Parameter(torch.randn(hidden_size,input_size))
        # Initialize weights for output transition from the last hidden layer
        self.W_y = nn.Parameter(torch.randn(input_size,hidden_size))
        # Initialize biases for hidden layers
        self.B_h = nn.Parameter(torch.zeros(num_layers,hidden_size))
        # Initialize bias for output layer
        self.B_y = nn.Parameter(torch.zeros(input_size))

        # If there are multiple layers, initialize weights for previous-layer-hidden to current-layer-hidden transitions
        if num_layers > 1:
            self.W_hh = nn.Parameter(torch.randn(num_layers - 1, hidden_size, hidden_size))
        else:
            self.W_hh = None # Handle the case of a single layer

        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forwardo(self,x):
        seq_len = x.shape[0]
        h = torch.zeros(self.num_layers,self.hidden_size,device=x.device)
        output = []
        for i in range(seq_len):
            h_new = torch.zeros_like(h)
            for layer in range(self.num_layers):
                if layer==0:
                    # Compute hidden state for the first layer
                    h_new[layer] = torch.tanh(self.W_h[layer]@h[layer]+self.W_x@x[i]+self.B_h[layer])
                else:
                    h_new[layer] = torch.tanh(self.W_h[layer]@h[layer]+self.W_hh[layer-1]@h_new[layer-1]+self.B_h[layer])
            h = h_new
            output.append(self.W_y @ h[-1] + self.B_y)
        return torch.stack(output),h
    
    def forward(self,x):
        seq_len = x.shape[0]
        h = torch.zeros(self.num_layers, self.hidden_size,device=x.device)
        output = []
        for i in range(seq_len):
            h_new = []
            for layer in range(self.num_layers):
                if layer==0:
                    h_layer = torch.tanh(self.W_h[layer]@h[layer]+self.W_x@x[i]+self.B_h[layer])
                else:
                    h_layer = torch.tanh(self.W_h[layer]@h[layer]+self.W_hh[layer-1]@h_new[layer-1]+self.B_h[layer])
                h_new.append(h_layer)
            h = torch.stack(h_new)
            output.append(self.W_y@h[-1]+self.B_y)
        return torch.stack(output),h
    
class GRUv2(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.Uu 
        self.Wu
        self.Bu

        self.Ur
        self.Wr 
        self.Br 

        self.U
        self.W 
        self.B 


class LSTMv2(nn.Module):

    def __init__(self,):
        super().__init__()
        

