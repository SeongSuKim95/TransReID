import torch
from torch import nn
import torch.nn.functional as F

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))

class KLD_mean(nn.Module):
    def __init__(self):
        super(KLD_mean, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.ID = 4
    def forward(self, p: torch.tensor, q: torch.tensor):
        p,q = p.log(), q.log()
        q = (q.reshape(-1,self.ID,q.shape[-2],q.shape[-1])).mean(dim=1).repeat_interleave(self.ID,dim=0)
        return self.kl(p,q)
        
class LayerWise_JSD(nn.Module):
    def __init__(self):
        super(LayerWise_JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.num_heads = 12
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        
        return 0.5 * (self.kl(p.log(),m) + self.kl(q.log(),m)) 