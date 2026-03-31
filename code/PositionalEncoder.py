import torch 
import torch.nn as nn
import math

class PostionalEncoder(nn.Module):
    def __init__(self , dim:int , max_len:int):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len , dim)
        ## unsuqeeze 使其能直接矩阵乘
        position = torch.arange(0,max_len,dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,dim,2).float() * -(math.log(10000)) / dim)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        #batchsize维
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self , x:torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:,:seq_len]

PE = PostionalEncoder(512,3)
input = torch.rand(3,2,512)
output = PE(input)
print(output.shape)
print(output)