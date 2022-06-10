from re import A
from tkinter import X
import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
import sys
import wandb
from .JSD_loss import JSD, LayerWise_JSD, KLD_mean
def normalize_max(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    dis = torch.sum(x.pow(2), dim=1).sqrt()
    m, _ = torch.max(dis, 0)
    x = x / m
    return x

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def patchwise_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [m, d]
    Returns:
      dist: pytorch Variable, with shape [m]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def hard_example_distance(dist_mat, labels):
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )
    # 
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
    )
    dist_an_mean = torch.mean(dist_mat[is_neg].contiguous().view(N, -1), dim=1)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return (
        dist_ap,
        dist_an,
        dist_an_mean,
        (
            is_pos,
            is_neg,
            relative_p_inds,
            relative_n_inds,
        ),
    )

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_example_mining_with_inds(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    (
        dist_ap,
        dist_an,
        dist_an_mean,
        (
            is_pos,
            is_neg,
            relative_p_inds,
            relative_n_inds,
        ),
    ) = hard_example_distance(dist_mat, labels)

    # shape [N, N]
    ind = (
        labels.new()
        .resize_as_(labels)
        .copy_(torch.arange(0, N).long())
        .unsqueeze(0)
        .expand(N, N)
    )
    # shape [N, 1]
    p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data) # relative_p_inds to global_inds
    n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)


    return dist_ap, dist_an, dist_an_mean, p_inds, n_inds

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, feat_norm, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        self.normalize_feature = feat_norm
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):

        if self.normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels) # hard batch mining

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
class TripletAttentionLoss_ss_pos_6(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, num_instance, rel_pos, comb, replacement, comb_idx, jsd, head_wise, head_num,margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
        self.LW_loss = LayerWise_JSD()
        self.comb = comb 
        self.comb_idx = comb_idx
        self.JSD = jsd
        self.HEAD_WISE = head_wise
        self.HEAD_NUM = head_num
        self.KLD_loss = nn.KLDivLoss(reduction='batchmean')
        self.KLD_mean = KLD_mean()
        self.REPLACEMENT = replacement

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(
        self,
        global_feat: torch.Tensor,
        labels: torch.Tensor,
        rel_pos_bias,
        abs_pos,
        cls_param,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:

        cls_feat = global_feat[:,0] # detach()
        dist_mat_cls = euclidean_dist(cls_feat,cls_feat)

        (
            dist_ap_cls,
            dist_an_cls,
            dist_an_mean_cls,
            ind_pos_cls,
            ind_neg_cls,
        ) = hard_example_mining_with_inds(dist_mat_cls, labels)

        patch_feat_A = global_feat[:,1:]
        B,N,C = patch_feat_A.shape
        ID = self.num_instance
        scale = cls_feat.shape[-1] ** 0.5
        param = cls_param[labels]

        param_sim= (((param.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        val_anc_param,ind_anc_param = torch.topk(param_sim,self.comb_idx,dim=-1)
        
        # ind_anc_param = ind_anc[:,:self.comb_idx]
        if self.comb :
            anc_comb = [torch.combinations(x,r=2,with_replacement=self.REPLACEMENT) for x in ind_anc_param]
        else :
            anc_comb = [torch.cartesian_prod(x,x) for x in ind_anc_param]

        anc_comb = torch.cat(anc_comb,0).reshape(B,-1,2)
        anc_rel = []
        anc_vec = []
        abs_pos = abs_pos[1:]
        for i in range(B):
            if self.rel_pos :
                    rel_val_anc = torch.stack([rel_pos_bias[:,x[0],x[1]] for x in anc_comb[i]])
                    abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                    abs_val_anc = abs_val_anc.unsqueeze(1).expand(rel_val_anc.size(0),rel_val_anc.size(1))
                    anc_vec.append(rel_val_anc+abs_val_anc)

            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)
        
        # Patch position vector
        anc_vec = (torch.cat(anc_vec).reshape(B,anc_comb.size(1),-1).transpose(-2,-1).softmax(-1)).reshape(B,-1)

        if self.comb :
            if self.HEAD_WISE:
                if self.REPLACEMENT:
                    anc_rel = anc_rel.reshape(B,-1,int((self.comb_idx * (self.comb_idx + 1))/2))
                else : 
                    anc_vec = anc_vec.reshape(B,-1,int((self.comb_idx * (self.comb_idx - 1))/2))
            else :
                if self.REPLACEMENT:
                    anc_vec = anc_vec.reshape(B,-1,int((self.HEAD_NUM * self.comb_idx * (self.comb_idx+1))/2))
                else :
                    anc_vec = anc_vec.reshape(B,-1,int((self.HEAD_NUM * self.comb_idx * (self.comb_idx - 1))/2))
        else :
            if self.HEAD_WISE:
                anc_vec = anc_vec.reshape(B,-1,self.comb_idx * self.comb_idx)
            else :
                anc_vec = anc_vec.reshape(B,-1,self.HEAD_NUM * self.comb_idx * self.comb_idx)
        pos_vec = anc_vec[ind_pos_cls]

        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)

        else:
            if self.JSD : 
                    if self.HEAD_WISE : 
                        position_loss_a = self.JSD_loss(anc_vec,pos_vec)
                    else :
                        position_loss_a = self.JSD_loss(anc_vec,pos_vec)

            else :
                position_loss_a = self.KLD_loss(anc_vec.log(),pos_vec.log())
            loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            loss = loss_cls + self.loss_ratio * position_loss_a
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

            return loss, loss_cls, position_loss_a, dist_ap_cls, dist_an_cls 
