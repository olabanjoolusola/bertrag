
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        logits_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        mask = mask & logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)+1e-12)
        loss = -(mask.float()*log_prob).sum(1)/(mask.float().sum(1)+1e-12)
        return loss.mean()
