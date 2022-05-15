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

class HeadWise_JSD(nn.Module):
    def __init__(self):
        super(HeadWise_JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.num_heads = 12
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, self.num_heads, p.size(-1)), q.view(-1, self.num_heads, q.size(-1))
        loss = 0
        for i in range(p.size(0)):
            p_temp,q_temp = p[i],q[i]
            m = (0.5 * (p_temp + p_temp)).log()
            loss += 0.5 * (self.kl(p_temp.log(),m) + self.kl(q_temp.log(),m)) 
        return loss 