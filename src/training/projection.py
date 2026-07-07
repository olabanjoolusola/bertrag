
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self,in_dim=768,hidden=512,out_dim=128,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden,out_dim)
        )
    def forward(self,x):
        return F.normalize(self.net(x),dim=1)
