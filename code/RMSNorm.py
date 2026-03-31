import torch
import torch.nn as nn
class RMSNorm(torch.nn.Module):
    def __init__(self , dim:int , eps: float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight  = nn.Parameter(torch.ones(dim))
    
    def __norm(self, x:torch.Tensor):
        return  torch.rsqrt(x.pow(2).mean(-1,keepdim = True) + self.eps)
    
    def forward(self , x:torch.Tensor):
        return self.weight*(x*self.__norm(x.float())).type_as(x)

RMS = RMSNorm(512)
x = torch.rand(2,3,512)
output = RMS(x)
print(output.shape)