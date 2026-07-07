
from transformers import AutoModel
import torch.nn as nn
from projection import ProjectionHead

class DomainBERT(nn.Module):
    def __init__(self,model_name="models/domainbert"):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_name)
        self.projector=ProjectionHead()

    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        cls=out.last_hidden_state[:,0]
        return self.projector(cls)
