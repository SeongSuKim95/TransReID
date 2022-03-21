import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
import sys
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
    # p_inds : the most far sample index among postive ID (global index)
    # n_inds : the most close sample index among negative ID (global index)

    return dist_ap, dist_an, dist_an_mean, p_inds, n_inds

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
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


class TripletAttentionLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin: Optional[float] = None):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.weight_param = nn.Parameter(
            torch.ones(1, dtype=torch.float, requires_grad=True).cuda()
        )

    def __call__(
        self,
        global_feat: torch.Tensor,
        labels: torch.Tensor,
        cls_param: torch.Tensor,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if normalize_feature:
            global_feat = normalize_max(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)

        #dist_mat = cosine_distance(global_feat,global_feat)
        ( 
            dist_ap,
            dist_an,
            dist_an_mean,
            ind_pos,
            ind_neg,
        ) = hard_example_mining_with_inds(dist_mat, labels)
        neg_weight = self.weight(ind_neg, cls_param.detach(), labels) # weight abs차의 normalize 값이 threshold 보다 작으면 0 
        # neg_weight --> BoT : [64,2048] , ViT : [64, 768]
        # global_feat --> [64,2048] 
        # Euclidean distance between Weighted feature of Anchor & negative, positive
        dist_neg = torch.sum(
            (global_feat * neg_weight - global_feat[ind_neg] * neg_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        dist_pos = torch.sum(
            (global_feat * neg_weight - global_feat[ind_pos] * neg_weight).pow(2), dim=1
        ).sqrt()
        y = dist_an.new().resize_as_(dist_an).fill_(1) # y.shape = 64

        if self.margin is not None:
            
            HTH = self.ranking_loss(dist_an.detach(), dist_ap, y)
            TH =  self.ranking_loss(dist_neg,dist_pos,y)
            HNTH_P2 = self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            
            # NEWTH
            Triplet_loss = HTH + TH + HNTH_P2
            
            # EWTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) + self.ranking_loss(
            #     dist_neg, dist_pos, y
            # ) 

            # HNTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) + self.ranking_loss(
            #     dist_an_mean, dist_ap.detach(), y
            # ) 

            # HTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) 
            
            # TH
            # loss = self.ranking_loss(dist_an, dist_ap, y) 
        else:
            Triplet_loss = (
                self.ranking_loss(dist_an.detach() - dist_ap, y)
                + self.ranking_loss(dist_neg - dist_pos, y)
                + self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            )
        if torch.isnan(Triplet_loss):
            sys.exit("Gradient Exploded")

        return Triplet_loss,HTH,TH,HNTH_P2 

    def weight(
        self, ind_neg: torch.Tensor, param: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor: # target = labels
        t = 0.1
        weight_neg1 = param[target] # 각 sample ID의 weight vector [64,768]
        weight_neg2 = param[target[ind_neg]] # 각 sample ID와 가장 먼 sample ID의 weight vector
        weight_neg = torch.abs(weight_neg1 - weight_neg2) # 둘의 차이, [64,2048]
        max, _ = torch.max(weight_neg, dim=1, keepdim=True) # max : [64,1] , weight_neg 각 행에서 가장 큰 값
        weight_neg = weight_neg / (max + 1e-12) # 큰값으로 normalize
        weight_neg[weight_neg < t] = -self.weight_param 
        weight_neg = weight_neg + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로

        return weight_neg
class TripletPatchAttentionLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin: Optional[float] = None):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.weight_param = nn.Parameter(
            torch.ones(1, dtype=torch.float, requires_grad=True).cuda()
        )

    def __call__(
        self,
        global_feat: torch.Tensor,
        labels: torch.Tensor,
        cls_param: torch.Tensor,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_feat = global_feat[:,0,:]

        if normalize_feature:
            cls_feat = normalize_max(cls_feat, axis=-1)
        dist_mat = euclidean_dist(cls_feat, cls_feat)

        #dist_mat = cosine_distance(global_feat,global_feat)
        (
            dist_ap,
            dist_an,
            dist_an_mean,
            ind_pos,
            ind_neg,
        ) = hard_example_mining_with_inds(dist_mat, labels)
        neg_weight = self.weight(ind_neg, global_feat.detach(), cls_param ,labels) # weight abs차의 normalize 값이 threshold 보다 작으면 0 
        # neg_weight --> BoT : [64,2048] , ViT : [64, 768]
        # global_feat --> [64,2048] 
        # Euclidean distance between Weighted feature of Anchor & negative, positive
        dist_neg = torch.sum(
            (global_feat * neg_weight - global_feat[ind_neg] * neg_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        dist_pos = torch.sum(
            (global_feat * neg_weight - global_feat[ind_pos] * neg_weight).pow(2), dim=1
        ).sqrt()
        y = dist_an.new().resize_as_(dist_an).fill_(1) # y.shape = 64

        if self.margin is not None:
            
            HTH = self.ranking_loss(dist_an.detach(), dist_ap, y)
            TH =  self.ranking_loss(dist_neg,dist_pos,y)
            HNTH_P2 = self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            
            # NEWTH
            Triplet_loss = HTH + TH + HNTH_P2
            
            # EWTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) + self.ranking_loss(
            #     dist_neg, dist_pos, y
            # ) 

            # HNTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) + self.ranking_loss(
            #     dist_an_mean, dist_ap.detach(), y
            # ) 

            # HTH
            # loss = self.ranking_loss(dist_an.detach(), dist_ap, y) 
            
            # TH
            # loss = self.ranking_loss(dist_an, dist_ap, y) 
        else:
            Triplet_loss = (
                self.ranking_loss(dist_an.detach() - dist_ap, y)
                + self.ranking_loss(dist_neg - dist_pos, y)
                + self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            )
        if torch.isnan(Triplet_loss):
            print("d")

        return Triplet_loss,HTH,TH,HNTH_P2 

    def weight(
        self, ind_neg: torch.Tensor, global_feat : torch.Tensor, param: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor: # target = labels
        t = 0.1
        weight_neg1 = param[target] # 각 sample ID의 weight vector [64,768]
        weight_neg2 = param[target[ind_neg]] # 각 sample ID와 가장 먼 sample ID의 weight vector
        weight_neg = torch.abs(weight_neg1 - weight_neg2) # 둘의 차이, [64,2048]
        max, _ = torch.max(weight_neg, dim=1, keepdim=True) # max : [64,1] , weight_neg 각 행에서 가장 큰 값
        weight_neg = weight_neg / (max + 1e-12) # 큰값으로 normalize
        weight_neg[weight_neg < t] = -self.weight_param 
        weight_neg = weight_neg + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로

        # weight_neg_patch1 = global_feat[target,1:,:]
        # weight_neg_patch2 = global_feat[target[ind_neg],1:,:]
        # test = torch.mul(weight_neg_patch1,weight_neg_patch2)
        # dist_mat = patchwise_dist(global_feat, global_feat)

        return weight_neg
