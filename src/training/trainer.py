
import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self,model,loss_fn,optimizer,device):
        self.model=model.to(device)
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.device=device
        self.scaler=GradScaler()

    def train_epoch(self,dataloader):
        self.model.train()
        total=0.0
        for batch in dataloader:
            labels=batch.pop("labels").to(self.device)
            batch={k:v.to(self.device) for k,v in batch.items()}
            self.optimizer.zero_grad()
            with autocast():
                feats=self.model(batch["input_ids"],batch["attention_mask"])
                loss=self.loss_fn(feats,labels)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total+=loss.item()
        return total/len(dataloader)
