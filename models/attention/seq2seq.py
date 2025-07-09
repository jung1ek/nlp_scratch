from torch import nn
import torch

class Encoder(nn.Module):

    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # input to hidden weight forward
        self.fw_ih = nn.Parameter(torch.randn(self.hidden_size,input_size))
        self.fb_ih = nn.Parameter(torch.randn(self.hidden_size))

        # hidden to hidden weight
        self.fw_hh = nn.Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.fb_hh = nn.Parameter(torch.randn(self.hidden_size))

        # input to hidden weight backward
        self.bw_ih = nn.Parameter(torch.randn(self.hidden_size,input_size))
        self.bb_ih = nn.Parameter(torch.randn(self.hidden_size))

        # hidden to hidden weight
        self.bw_hh = nn.Parameter(torch.randn(self.hidden_size,self.hidden_size))
        self.bb_hh = nn.Parameter(torch.randn(self.hidden_size))
    
    def forward(self,x):
        seq_len,_ = x.shape
        f_h0 = torch.zeros(self.hidden_size)
        b_h0 = torch.zeros(self.hidden_size)
        f_h = f_h0
        b_h = b_h0
        f_output = []
        b_output = []
        for j in range(seq_len):
            f_h = torch.tanh((self.fw_hh@f_h+self.fb_hh) + (self.fw_ih@x[j]+self.fb_ih))
            b_h = torch.tanh((self.bw_hh@b_h+self.bb_hh) + (self.bw_ih@x[-1-j]+self.bb_ih))
            
            f_output.append(f_h)
            b_output.append(b_h)
        #concat
        return torch.cat([torch.stack(f_output),reversed(torch.stack(b_output))],dim=-1)
    

class Decoder(nn.Module):

    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # decoder hidden weight, for calc e.
        self.w = nn.Parameter(torch.randn(hidden_size,hidden_size))

        # encoder's hidden weight, for calc e.
        self.u = nn.Parameter(torch.randn(self.hidden_size,self.hidden_size*2))

        # ffn weight
        self.v = nn.Parameter(torch.randn(hidden_size))

        # weghts for decoder calculating hidden value (s),c_s, candidate s
        self.w_w = nn.Parameter(torch.randn(hidden_size,input_size))
        self.w_u = nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.w_c = nn.Parameter(torch.randn(hidden_size,hidden_size))

        # reset gate and update gate.
        self.w_z = nn.Parameter(torch.randn(hidden_size,input_size))
        self.w_r = nn.Parameter(torch.randn(hidden_size,input_size))
        self.u_z = nn.Parameter(torch.randn(hidden_size,input_size))
        self.u_r = nn.Parameter(torch.randn(hidden_size,input_size))
        self.c_z = nn.Parameter(torch.randn(hidden_size,input_size*2))
        self.c_r = nn.Parameter(torch.randn(hidden_size,input_size*2))

    def forward(self,enc_h,y):
        seq_len,_= y.shape
        s = torch.zeros(self.hidden_size)

        # calculating attention, and making each input scalar. (seq,h_enc) = seq,
        # eij refers to for each i in output seq, every j (input seq), matrix correlation of each input with output
        for i in range(seq_len):
            ei =  torch.tanh(self.w@s + enc_h @ self.u.T) @ self.v
            # weight vector of dim(input_seq,)
            ai = torch.exp(ei)/torch.sum(torch.exp(ei),dim=-1)
            # weighted sum of input_seq ,context vector of dim 2*n
            ci = ai @ enc_h
            # reset gate
            r = torch.sigmoid(self.w_r@y[i] + self.u_r@s + self.c_r@ci)
            # update gate
            z = torch.sigmoid(self.w_z@y[i] + self.u_z@s + self.c_z@ci)
            # new candidate s
            c_s = torch.tanh(self.w_w @ y[i] + self.w_u@(s*r) + self.w_c @ ci )

            # new state s, if z~1 keep new info mostly, if 0 old info mostly
            s = (1-z) * s + z*c_s
            
            

            





        