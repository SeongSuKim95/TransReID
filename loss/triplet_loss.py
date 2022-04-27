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
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        cls_feat = global_feat[:,0]
        
        patch_feat = global_feat[:,1:]
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
        
        neg_weight = self.weight(ind_neg, patch_feat.detach()) # weight abs차의 normalize 값이 threshold 보다 작으면 0 
        # neg_weight --> BoT : [64,2048] , ViT : [64, 768]
        # global_feat --> [64,2048] 
        # Euclidean distance between Weighted feature of Anchor & negative, positive
        anchor_feat = torch.mean(patch_feat * neg_weight, dim=1)
        positive_feat = torch.mean(patch_feat[ind_pos] * neg_weight, dim=1)
        negative_feat = torch.mean(patch_feat[ind_neg] * neg_weight, dim=1)

        dist_neg = torch.norm(anchor_feat - negative_feat, p=2, dim=1)# * : element wise multiplication
        dist_pos = torch.norm(anchor_feat - positive_feat, p=2, dim=1)

        y = dist_an.new().resize_as_(dist_an).fill_(1) # y.shape = 64

        if self.margin is not None:
            
            # HTH = self.ranking_loss(dist_an.detach(), dist_ap, y)
            # TH =  self.ranking_loss(dist_neg,dist_pos,y)
            # HNTH_P2 = self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            # Triplet_loss = HTH + TH + HNTH_P2

            # NEWTH
            # loss = HTH + TH + HNTH_P2
            
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
            Triplet_loss = self.ranking_loss(dist_neg, dist_pos, y) 
            
        else:
            Triplet_loss = self.ranking_loss(dist_neg - dist_pos, y) 

            # Triplet_loss = (
            #     self.ranking_loss(dist_an.detach() - dist_ap, y)
            #     + self.ranking_loss(dist_neg - dist_pos, y)
            #     + self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
            # )

        return Triplet_loss

    def weight(
        self, ind_neg: torch.Tensor, global_feat : torch.Tensor
    ) -> torch.Tensor: # target = labels

        # weight_neg1 = param[target] # 각 sample ID의 weight vector [64,768]
        # weight_neg2 = param[target[ind_neg]] # 각 sample ID와 가장 먼 sample ID의 weight vector
        # weight_neg = torch.abs(weight_neg1 - weight_neg2) # 둘의 차이, [64,2048]
        # max, _ = torch.max(weight_neg, dim=1, keepdim=True) # max : [64,1] , weight_neg 각 행에서 가장 큰 값
        # weight_neg = weight_neg / (max + 1e-12) # 큰값으로 normalize
        # weight_neg[weight_neg < t] = -self.weight_param 
        # weight_neg = weight_neg + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        
        weight_neg_patch1 = global_feat # 각 sample ID의 patch feature 
        weight_neg_patch2 = global_feat[ind_neg] # 각 sample ID와 가장 먼 sample ID의 patch 
        
        # cos_distance_weight

        dot= torch.sum(weight_neg_patch1 * weight_neg_patch2,dim=-1)
        weight_neg_patch1_norm = torch.norm(weight_neg_patch1, p=2, dim=-1)
        weight_neg_patch2_norm = torch.norm(weight_neg_patch2, p=2, dim=-1)
        cos = dot/(weight_neg_patch1_norm * weight_neg_patch2_norm)
        acos = torch.acos(cos)
        acos_max, _ = torch.max(acos,dim=1,keepdim=True)
        acos_norm = acos / (acos_max + 1e-12)
        neg_weight = acos_norm.unsqueeze(-1)
        
        # Need to fix

        # Euclidean_distance_weight
        dist = torch.norm(torch.abs(weight_neg_patch1 - weight_neg_patch2),p=2,dim=-1)
        max, _ = torch.max(dist, dim = 1 ,keepdim = True)
        dist  = dist / (max + 1e-12)
        _, idx = dist.sort(dim=1)
        dist[idx<idx.shape[1]*0.5] = -self.weight_param
        dist = dist + self.weight_param
        # dist = dist - self.weight_param
        neg_weight = dist.unsqueeze(-1)
        # For numerical stability
        # 나중에 해보기
        # acos_norm[acos_norm<t] = -self.weight_param
        # acos_norm = acos_norm + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        return neg_weight
class TripletBranchLoss(object):
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

    def __call__(
        self,
        triplet_feat: torch.Tensor,
        labels: torch.Tensor,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        
        Anchor_P = triplet_feat[0]
        Anchor_N = triplet_feat[1]
        Positive = triplet_feat[2]
        Negative = triplet_feat[3] # Anchor의 scale이 더 큰 상황

        #dist_mat = cosine_distance(global_feat,global_feat) 

        dist_an = torch.norm((Anchor_N-Negative),p=2,dim=1)
        dist_ap = torch.norm((Anchor_P-Positive),p=2,dim=1)

        y = dist_an.new().resize_as_(dist_an).fill_(1) # y.shape = 64

        if self.margin is not None:
            Triplet_loss = self.ranking_loss(dist_an, dist_ap, y) 
        else:
            Triplet_loss = self.ranking_loss(dist_an - dist_ap, y) 

        return Triplet_loss, dist_ap, dist_an
class TripletBranchLoss_1(object):
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

    def __call__(
        self,
        triplet_feat: torch.Tensor,
        p_inds: torch.Tensor,
        n_inds: torch.Tensor,
        labels: torch.Tensor,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        
        Anchor = triplet_feat
        Negative = triplet_feat[n_inds]
        Positive = triplet_feat[p_inds]
        
        #dist_mat = cosine_distance(global_feat,global_feat) 

        dist_an = torch.norm((Anchor-Negative),p=2,dim=1)
        dist_ap = torch.norm((Anchor-Positive),p=2,dim=1)

        y = dist_an.new().resize_as_(dist_an).fill_(1) # y.shape = 64

        if self.margin is not None:
            Triplet_loss = self.ranking_loss(dist_an, dist_ap, y) 
        else:
            Triplet_loss = self.ranking_loss(dist_an - dist_ap, y) 

        return Triplet_loss, dist_ap, dist_an
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

class TripletAttentionLoss_ss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.weight_param = nn.Parameter(
            torch.ones(1, dtype=torch.float, requires_grad=True).cuda()
        )
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(
        self,
        global_feat: torch.Tensor,
        labels: torch.Tensor,
        epoch,
        cls_param: torch.Tensor,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0.1
        # if epoch >= self.max_epoch * 0.5 :
        #     patch_ratio = self.patch_ratio * 0.5
        # else :
        #     patch_ratio = self.patch_ratio
        cls_feat = global_feat[:,0] # detach()
        
        # cls_feat_detach = global_feat[:,0].detach()
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
        #################################################        
        # if normalize_feature:
        #     cls_feat = normalize_max(cls_feat, axis=-1)
        # dist_mat = euclidean_dist(cls_feat, cls_feat)
        # cls_similarity = (cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1)/scale
        # cls_similarity = cls_similarity.softmax(-1)
        # dist_mat = cosine_distance(global_feat,global_feat)
        #################################################
        
        # Method 5
        pos_cls = cls_feat[ind_pos_cls]
        neg_cls = cls_feat[ind_neg_cls]

        anc_sim = (cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1)
        pos_sim = (pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1)
        neg_sim = (neg_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1)
        # Performance Index 242
        # pos_sim = anc_sim[ind_pos_cls]
        # neg_sim = anc_sim[ind_neg_cls]

        rank = int(N* self.patch_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = val_anc[ind_pos_cls], ind_anc[ind_pos_cls]
        val_neg, ind_neg = val_anc[ind_neg_cls], ind_anc[ind_neg_cls]
      
        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)
        cat_neg = torch.cat((ind_anc,ind_neg),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]
       
        cat_neg_idx, cat_neg_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_neg],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_neg]
        intersect_neg = [cat_neg_idx[i][cat_neg_cnts[i]!=1] for i in range(B)]
        patches_neg = [patch_feat_A[i][intersect_neg[i]] for i in range(B)]
        
        anc_pos_val = [anc_sim[i][intersect_pos[i]]/anc_sim[i][intersect_pos[i]].max() for i in range(B)]
        anc_neg_val = [anc_sim[i][intersect_neg[i]]/anc_sim[i][intersect_neg[i]].max() for i in range(B)]
         
        pos_val = [pos_sim[i][intersect_pos[i]]/pos_sim[i][intersect_pos[i]].max() for i in range(B)]
        neg_val = [neg_sim[i][intersect_neg[i]]/neg_sim[i][intersect_neg[i]].max() for i in range(B)]

        anc_pos_weighted_patches = torch.stack([torch.mean(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)])
        anc_neg_weighted_patches = torch.stack([torch.mean(patches_neg[i]*anc_neg_val[i].unsqueeze(-1),dim=0) for i in range(B)])
        pos_weighted_patches = torch.stack([torch.mean(patches_pos[i]*pos_val[i].unsqueeze(-1),dim=0) for i in range(B)])
        neg_weighted_patches = torch.stack([torch.mean(patches_neg[i]*neg_val[i].unsqueeze(-1),dim=0) for i in range(B)])
        
        cls_feat_detach = cls_feat
        anc_pos_detach = anc_pos_weighted_patches
        anc_neg_detach = anc_neg_weighted_patches
        pos_detach = pos_weighted_patches
        neg_detach = neg_weighted_patches

        cls_norm = torch.norm(cls_feat_detach,p=2,dim=1)
        anc_pos_norm = torch.norm(anc_pos_detach,p=2,dim=1)
        anc_neg_norm = torch.norm(anc_neg_detach,p=2,dim=1)
        pos_norm = torch.norm(pos_detach,p=2,dim=1)
        neg_norm = torch.norm(neg_detach,p=2,dim=1)

        anc_pos_norm_ratio = (cls_norm / anc_pos_norm).unsqueeze(-1)
        anc_neg_norm_ratio = (cls_norm / anc_neg_norm).unsqueeze(-1)
        pos_norm_ratio = (cls_norm[ind_pos_cls] / pos_norm).unsqueeze(-1)
        neg_norm_ratio = (cls_norm[ind_neg_cls] / neg_norm).unsqueeze(-1)

        anc_pos_diff = (cls_feat - anc_pos_weighted_patches * anc_pos_norm_ratio)
        anc_neg_diff = (cls_feat - anc_neg_weighted_patches * anc_neg_norm_ratio)
        pos_diff = (pos_cls - pos_weighted_patches * pos_norm_ratio)
        neg_diff = (neg_cls - neg_weighted_patches * neg_norm_ratio)

        abs = torch.abs(torch.cat((anc_pos_diff,anc_neg_diff,pos_diff,neg_diff)))
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        # abs_norm = (abs / (abs_max+1e-12)).reshape(-1,B,C)
        # anc_pos_weight,anc_neg_weight,pos_weight,neg_weight = 1 - abs_norm[0],abs_norm[1],1 - abs_norm[2],abs_norm[3]
        # abs = torch.cat((anc_pos_weight,anc_neg_weight,pos_weight,neg_weight))
        
        abs_diff = abs / (abs_max + 1e-12)
        abs_common = 1 - abs/ (abs_max + 1e-12)
        abs_diff[abs_diff < t] = -self.weight_param 
        abs_common[abs_common<t] = -self.weight_param
        abs_diff = abs_diff + self.weight_param
        abs_common = abs_common + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        abs_diff = abs_diff.reshape(-1,B,C)
        abs_common = abs_common.reshape(-1,B,C)
        
        anc_pos_weight , anc_neg_weight, pos_weight, neg_weight = abs_diff[0],abs_diff[1],abs_diff[2],abs_diff[3]
        anc_pos_weight_c, anc_neg_weight_c, pos_weight_c, neg_weight_c = abs_common[0],abs_common[1],abs_common[2],abs_common[3]
        
        dist_pos = torch.sum(
            (cls_feat * anc_pos_weight - pos_cls * pos_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_neg_weight - neg_cls * neg_weight ).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
        dist_pos_common = torch.sum(
            (cls_feat * anc_pos_weight_c - pos_cls * pos_weight_c ).pow(2), dim=1
        ).sqrt()

        dist_neg_common = torch.sum(
            (cls_feat * anc_neg_weight_c - neg_cls * neg_weight_c ).pow(2), dim=1
        ).sqrt()
        #################################################
        # Method 1,2 

        # cls_feat_b = cls_feat.expand(ID,B,C).reshape((ID,-1,ID,C)).transpose(0,1).reshape(-1,C)
        # patch_feat_A_b = patch_feat_A.expand(ID,B,N,C).transpose(0,1).reshape(-1,N,C)
        # cls_similarity_b = (cls_feat_b.unsqueeze(1) @ patch_feat_A_b.transpose(-1,-2)).squeeze(1)/scale
        # rank = int(N*self.patch_ratio)
        # val,idx = torch.topk(cls_similarity_b.reshape(B,-1,N),rank,dim=-1)

        # # Method 4
        # # rank = int(N*self.patch_ratio)
        # # cls_similarity_b = cls_similarity_b.reshape(B,ID,N).sum(dim=1)
        # # val,idx = torch.topk(cls_similarity_b,rank,dim=1)
        # # dummy_idx = idx.unsqueeze(2).expand(idx.size(0),idx.size(1),patch_feat_A.size(2))
        # # out = patch_feat_A.gather(1,dummy_idx)
        # # max_val, _ = torch.max(val,dim=-1,keepdim=True)
        # # val = val / (max_val + 1e-12)
        
        # # val[val< t] = -self.weight_param  
        # # val = val + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        
        # # # val = val.softmax(-1)
        # # out = torch.mean((out*val.unsqueeze(-1)),dim=1)
       
        # # Method 1-1
        # val = val.reshape(B,-1)
        # idx = idx.reshape(B,-1)
        # dummy_idx = idx.unsqueeze(2).expand(idx.size(0),idx.size(1),patch_feat_A.size(2))
        # out = patch_feat_A.gather(1,dummy_idx) # gather corresponding patch
        # max , _ = torch.max(val,dim=1,keepdim=True)
        # val = val / (max + 1e-12)
        # out = torch.mean((out * val.unsqueeze(-1)),dim=1) # weighted summation of gathered patch
        
        # # Method 1-2
        # cls_feat_detach = cls_feat.detach()
        # out_detach = out.detach()
        # cls_norm = torch.norm(cls_feat_detach,p=2,dim=1)
        # out_norm = torch.norm(out_detach,p=2,dim=1)

        # norm_ratio = (cls_norm / out_norm).unsqueeze(-1)
        
        # diff = (cls_feat - out * norm_ratio)

        # abs = torch.abs(diff)
        # abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        # abs = 1 -(abs / (abs_max + 1e-12))
        # abs[abs < t] = -self.weight_param 
        # abs = abs + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        # dist_neg = torch.sum(
        #     (cls_feat * abs - cls_feat[ind_neg_cls] * abs ).pow(2), dim=1
        # ).sqrt() # * : element wise multiplication
        # dist_pos = torch.sum(
        #     (cls_feat * abs  - cls_feat[ind_pos_cls] * abs).pow(2), dim=1
        # ).sqrt()
        ################################################

        # Method 2
        # val = val.squeeze(-1)
        # max_val, _ = torch.max(val,dim=-1,keepdim=True)
        # max = max_val.squeeze(-1).sum(-1).unsqueeze(-1)
        # val = val.reshape(B,-1)/(max + 1e-12)
        # out = torch.mean((out * val.unsqueeze(-1)),dim=1)
        
        # Method 3
        # Anchor_rank = int(N*self.patch_ratio*2)
        # Positive_rank = int(N*self.patch_ratio)
        # mask = torch.zeros(ID*ID,ID*ID,dtype=torch.bool)
        # index = torch.arange(0,B//ID,ID+1)
        # mask[:,index] = True
        # mask = mask.view(-1)
        # Anchor_similarity = cls_similarity_b[mask]
        # Positive_similarity = cls_similarity_b[~mask]
        # anchor_val,anchor_idx = torch.topk(Anchor_similarity,Anchor_rank,dim=-1)
        # positive_val, positive_idx = torch.topk(Positive_similarity,Positive_rank,dim=-1)
        # positive_val = positive_val.reshape(B,-1)
        # positive_idx = positive_idx.reshape(B,-1)

        # for a_val, a_idx, p_val, p_idx in zip(anchor_val, anchor_idx, positive_val, positive_idx):
        #     p_intersect = p_idx.unique()
        #     cat = torch.cat((a_idx,p_intersect))
        #     cat_idx, cnts = cat.unique(return_counts=True)
        #     anchor_intersect = cat_idx[cnts!=1]
        
        # weight_param = torch.zeros((B*ID,N), dtype=cls_similarity_b.dtype, requires_grad=True).cuda()
        # idx = idx.reshape(-1,idx.shape[-1])
        # weight = weight_param.scatter_add_(1,idx,cls_similarity_b)
        # weight_sum = weight.reshape(B,-1,N).sum(dim=1)
        # # val_reshape = val.reshape(B,-1)
        # idx_reshape = idx.reshape(B,-1)

        # CONCAT with CLS token
        # out = torch.cat((cls_feat,out),dim=1)

        # dist_mat = euclidean_dist(out,out)
        # dist_ap, dist_an = hard_example_mining(dist_mat, labels) # hard batch mining
        # dist_ap *= (1.0 + self.hard_factor)
        # dist_an *= (1.0 - self.hard_factor)

        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        # dist_ap = torch.gather(dist_mat,1,ind_pos_cls.unsqueeze(-1))
        # dist_an = torch.gather(dist_mat,1,ind_neg_cls.unsqueeze(-1))
        # dist_ap = dist_ap.squeeze(1)
        # dist_an = dist_an.squeeze(1)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_weighted_common + loss_cls       
        return loss, dist_ap_cls, dist_an_cls
    # Triplet_loss = (
    #     self.ranking_loss(dist_an.detach() - dist_ap, y)
    #     + self.ranking_loss(dist_neg - dist_pos, y)
    #     + self.ranking_loss(dist_an_mean, dist_ap.detach(), y)
    # )

    # def weight(
    #     self, ind_neg: torch.Tensor, global_feat : torch.Tensor
    # ) -> torch.Tensor: # target = labels
    # Cross Attention loss

    # (
    #     dist_ap,
    #     dist_an,
    #     dist_an_mean,
    #     ind_pos,
    #     ind_neg,
    # ) = hard_example_mining_with_inds(dist_mat, labels)

    # patch_feat_P = patch_feat_A[ind_pos]
    # patch_feat_N = patch_feat_A[ind_neg]

    # AP_similarity = patch_feat_A @ patch_feat_P.transpose(-2,-1)
    # AN_similarity = patch_feat_A @ patch_feat_N.transpose(-2,-1)
    
    # AP_value, AP_index = torch.max(AP_similarity,-1)
    # AN_value, AN_index = torch.max(AN_similarity,-1)

    # AP_index = AP_index.unsqueeze(-1).expand(B,N,C)
    # AP_patch = torch.gather(patch_feat_P,1,AP_index) # Anchor의 patch와 가장 가까운 Positive patch
    
    # AN_index = AN_index.unsqueeze(-1).expand(B,N,C)
    # AN_patch = torch.gather(patch_feat_N,1,AN_index) # Anchor의 patch와 가장 가까운 Negative patch
    
    # #AP_dist = torch.norm(patch_feat_A - AP_patch,p=2,dim=2)
    # AP_dist = torch.cdist(patch_feat_A,AP_patch).diagonal(dim1=-2,dim2=-1)
    # AP_dist = torch.sum(AP_dist,dim=1)
    # #weighted_AP_dist = torch.sum(cls_similarity * AP_dist,dim=1)

    # #AN_dist = torch.norm(patch_feat_A - AN_patch,p=2,dim=2)
    # AN_dist = torch.cdist(patch_feat_A,AN_patch).diagonal(dim1=-2,dim2=-1)
    # AN_dist = torch.sum(AN_dist,dim=1)
    # #weighted_AN_dist = torch.sum(cls_similarity * AN_dist,dim=1)

    # # neg_weight --> BoT : [64,2048] , ViT : [64, 768]
    # # global_feat --> [64,2048] 
    # # Euclidean distance between Weighted feature of Anchor & negative, positive
    
    # #=================================================================================

    # # y = weighted_AN_dist.new().resize_as_(weighted_AN_dist).fill_(1) # y.shape = 64

    # # if self.margin is not None:
    # #     loss_ss = self.ranking_loss(weighted_AN_dist, weighted_AP_dist, y)
    # #     loss_cls = self.ranking_loss(dist_an, dist_ap,y)
    # #     Triplet_loss = loss_ss + loss_cls
    # # else:
    # #     loss_ss = self.ranking_loss(weighted_AN_dist - weighted_AP_dist, y)
    # #     loss_cls = self.ranking_loss(dist_an, dist_ap,y)
    # #     Triplet_loss = loss_ss + loss_cls
    # # return Triplet_loss, weighted_AN_dist, weighted_AP_dist
    # #=================================================================================
    # y = AN_dist.new().resize_as_(AN_dist).fill_(1) # y.shape = 64

    # if self.margin is not None:
    #     loss_ss = self.ranking_loss(AN_dist, AP_dist, y)
    #     #loss_cls = self.ranking_loss(dist_an, dist_ap,y)
    #     Triplet_loss = loss_ss 
    # else:
    #     loss_ss = self.ranking_loss(AN_dist - AP_dist, y)
    #     #loss_cls = self.ranking_loss(dist_an, dist_ap,y)
    #     Triplet_loss = loss_ss
    # return Triplet_loss, AN_dist, AP_dist
    # def weight(
    #     self, ind_neg: torch.Tensor, global_feat : torch.Tensor
    # ) -> torch.Tensor: # target = labels

    #     # weight_neg1 = param[target] # 각 sample ID의 weight vector [64,768]
    #     # weight_neg2 = param[target[ind_neg]] # 각 sample ID와 가장 먼 sample ID의 weight vector
    #     # weight_neg = torch.abs(weight_neg1 - weight_neg2) # 둘의 차이, [64,2048]
    #     # max, _ = torch.max(weight_neg, dim=1, keepdim=True) # max : [64,1] , weight_neg 각 행에서 가장 큰 값
    #     # weight_neg = weight_neg / (max + 1e-12) # 큰값으로 normalize
    #     # weight_neg[weight_neg < t] = -self.weight_param 
    #     # weight_neg = weight_neg + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        
    #     weight_neg_patch1 = global_feat # 각 sample ID의 patch feature 
    #     weight_neg_patch2 = global_feat[ind_neg] # 각 sample ID와 가장 먼 sample ID의 patch 
        
    #     # cos_distance_weight

    #     dot= torch.sum(weight_neg_patch1 * weight_neg_patch2,dim=-1)
    #     weight_neg_patch1_norm = torch.norm(weight_neg_patch1, p=2, dim=-1)
    #     weight_neg_patch2_norm = torch.norm(weight_neg_patch2, p=2, dim=-1)
    #     cos = dot/(weight_neg_patch1_norm * weight_neg_patch2_norm)
    #     acos = torch.acos(cos)
    #     acos_max, _ = torch.max(acos,dim=1,keepdim=True)
    #     acos_norm = acos / (acos_max + 1e-12)
    #     neg_weight = acos_norm.unsqueeze(-1)
        
    #     # Need to fix

    #     # Euclidean_distance_weight
    #     dist = torch.norm(torch.abs(weight_neg_patch1 - weight_neg_patch2),p=2,dim=-1)
    #     max, _ = torch.max(dist, dim = 1 ,keepdim = True)
    #     dist  = dist / (max + 1e-12)
    #     _, idx = dist.sort(dim=1)
    #     dist[idx<idx.shape[1]*0.5] = -self.weight_param
    #     dist = dist + self.weight_param
    #     # dist = dist - self.weight_param
    #     neg_weight = dist.unsqueeze(-1)
    #     # For numerical stability
    #     # 나중에 해보기
    #     # acos_norm[acos_norm<t] = -self.weight_param
    #     # acos_norm = acos_norm + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
    #     return neg_weight
