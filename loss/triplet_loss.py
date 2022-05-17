from re import A
import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
import sys
import wandb
from .JSD_loss import JSD, LayerWise_JSD
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
class TripletAttentionLoss_ss_pos_2(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
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
        rel_pos_bias,
        abs_pos,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
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

        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        pos_sim = (((pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)
        
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)
        
        # Performance Index 242
        # pos_sim = anc_sim[ind_pos_cls]
        # neg_sim = anc_sim[ind_neg_cls]
        
        # rank = int(N* self.patch_ratio)
        # p_ratio = ((self.patch_ratio[1]-self.patch_ratio[0])/(self.max_epoch-1))*(epoch-1) + self.patch_ratio[0]
        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = torch.topk(pos_sim,rank,dim=-1)
        
        ind_anc_5 = ind_anc[:,:5]

        anc_comb = [torch.combinations(x,r=2) for x in ind_anc_5]

        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)

        anc_vec = (torch.cat(anc_vec).reshape(B,-1)).softmax(-1)
        pos_vec = anc_vec[ind_pos_cls]
        #neg_vec = anc_vec[ind_neg_cls]
        
        JSD_loss = self.JSD_loss(anc_vec,pos_vec)
        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]
         
        # anc_pos_val = [anc_sim[i][intersect_pos[i]]/anc_sim[i][intersect_pos[i]].max()for i in range(B)]
        # anc_neg_val = [anc_sim[i][intersect_neg[i]]/anc_sim[i][intersect_neg[i]].max() for i in range(B)]
         
        # pos_val = [pos_sim[i][intersect_pos[i]]/pos_sim[i][intersect_pos[i]].max() for i in range(B)]
        # neg_val = [neg_sim[i][intersect_neg[i]]/neg_sim[i][intersect_neg[i]].max() for i in range(B)]

        anc_pos_val = [anc_sim[i][intersect_pos[i]] for i in range(B)]
        
        # cat_all = torch.cat((ind_pos,ind_neg,ind_anc),dim=-1)
        # cat_all_idx, cat_all_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_all],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_all]
        
        # intersect_all = [cat_all_idx[i][cat_all_cnts[i]==3] for i in range(B)]
        # patches_all = [patch_feat_A[i][intersect_all[i]] for i in range(B)]
        # anc_all_val = [anc_sim[i][intersect_all[i]]/anc_sim[i][intersect_all[i]].max() for i in range(B)]
        
        anc_pos_weighted_patches = torch.stack([torch.sum(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)]) # Anc Pos common patch - for Anc
        
        # cls_feat_detach = cls_feat
        # anc_pos_detach = anc_pos_weighted_patches
        # anc_neg_detach = anc_neg_weighted_patches
        # pos_detach = pos_weighted_patches
        # neg_detach = neg_weighted_patches

        # cls_norm = torch.norm(cls_feat_detach,p=2,dim=1)
        # anc_pos_norm = torch.norm(anc_pos_weighted,p=2,dim=1)
        # anc_neg_norm = torch.norm(anc_neg_weighted,p=2,dim=1)
        # pos_norm = torch.norm(pos_detach,p=2,dim=1)
        # neg_norm = torch.norm(neg_detach,p=2,dim=1)

        # anc_pos_norm_ratio = (cls_norm / anc_pos_norm).unsqueeze(-1)
        # anc_neg_norm_ratio = (cls_norm / anc_neg_norm).unsqueeze(-1)
        # pos_norm_ratio = (cls_norm[ind_pos_cls] / pos_norm).unsqueeze(-1)
        # neg_norm_ratio = (cls_norm[ind_neg_cls] / neg_norm).unsqueeze(-1)

        anc_diff = (cls_feat - anc_pos_weighted_patches * anc_ratio)
        abs = torch.abs(anc_diff)
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        abs_norm = (abs / (abs_max+1e-12))
        abs_common = 1 - abs_norm

        # anc_pos_weight,anc_neg_weight,pos_weight,neg_weight = 1 - abs_norm[0],abs_norm[1],1 - abs_norm[2],abs_norm[3]
        # abs = torch.cat((anc_pos_weight,anc_neg_weight,pos_weight,neg_weight))
        abs_norm[abs_norm<t] = -self.weight_param
        #abs_common[abs_common<t] = -self.weight_param
        
        abs_norm = abs_norm + self.weight_param
        #abs_common = abs_norm + self.weight_param
        anc_weight = abs_norm
        anc_common_weight = abs_common

        #abs_diff = abs / (abs_max + 1e-12)
        #abs_common = 1 - abs/ (abs_max + 1e-12)
        #abs_diff[abs_diff < 
        # t] = -self.weight_param 
        #abs_common[abs_common<t] = -self.weight_param
        #abs_diff = abs_diff + self.weight_param
        #abs_common = abs_common + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        #abs_diff = abs_diff.reshape(-1,B,C)
        #abs_common = abs_common.reshape(-1,B,C)
        
        #anc_pos_weight , anc_neg_weight, pos_weight, neg_weight = abs_diff[0],abs_diff[1],abs_diff[2],abs_diff[3]
        #anc_pos_weight_c, anc_neg_weight_c, pos_weight_c, neg_weight_c = abs_common[0],abs_common[1],abs_common[2],abs_common[3]
        
        dist_pos = torch.sum(
            (cls_feat * anc_weight - pos_cls * anc_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_weight - neg_cls * anc_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
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
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            #loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_dist = self.ranking_loss(dist_position_neg - dist_position_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_mean + loss_cls_weighted + JSD_loss
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls
class TripletAttentionLoss_ss_pos_5(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
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
        rel_pos_bias,
        abs_pos,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
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

        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        pos_sim = (((pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)
        
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)
        
        # Performance Index 242
        # pos_sim = anc_sim[ind_pos_cls]
        # neg_sim = anc_sim[ind_neg_cls]
        
        # rank = int(N* self.patch_ratio)
        # p_ratio = ((self.patch_ratio[1]-self.patch_ratio[0])/(self.max_epoch-1))*(epoch-1) + self.patch_ratio[0]
        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = torch.topk(pos_sim,rank,dim=-1)
        
        ind_anc_5 = ind_anc[:,:10]

        anc_comb = [torch.combinations(x,r=2) for x in ind_anc_5]
        anc_comb = torch.cat(anc_comb,0).reshape(B,-1,2)
        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[:,x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                abs_val_anc = abs_val_anc.unsqueeze(1).expand(rel_val_anc.size(0),rel_val_anc.size(1))
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)
        anc_vec = (torch.cat(anc_vec).reshape(B,anc_comb.size(1),-1).transpose(-2,-1).softmax(-1)).reshape(B,-1)
        pos_vec = anc_vec[ind_pos_cls]
        JSD_loss = self.JSD_loss(anc_vec,pos_vec)

        #neg_vec = anc_vec[ind_neg_cls]
        
        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]
         
        # anc_pos_val = [anc_sim[i][intersect_pos[i]]/anc_sim[i][intersect_pos[i]].max()for i in range(B)]
        # anc_neg_val = [anc_sim[i][intersect_neg[i]]/anc_sim[i][intersect_neg[i]].max() for i in range(B)]
         
        # pos_val = [pos_sim[i][intersect_pos[i]]/pos_sim[i][intersect_pos[i]].max() for i in range(B)]
        # neg_val = [neg_sim[i][intersect_neg[i]]/neg_sim[i][intersect_neg[i]].max() for i in range(B)]

        anc_pos_val = [anc_sim[i][intersect_pos[i]] for i in range(B)]
        
        # cat_all = torch.cat((ind_pos,ind_neg,ind_anc),dim=-1)
        # cat_all_idx, cat_all_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_all],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_all]
        
        # intersect_all = [cat_all_idx[i][cat_all_cnts[i]==3] for i in range(B)]
        # patches_all = [patch_feat_A[i][intersect_all[i]] for i in range(B)]
        # anc_all_val = [anc_sim[i][intersect_all[i]]/anc_sim[i][intersect_all[i]].max() for i in range(B)]
        
        anc_pos_weighted_patches = torch.stack([torch.sum(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)]) # Anc Pos common patch - for Anc
        
        # cls_feat_detach = cls_feat
        # anc_pos_detach = anc_pos_weighted_patches
        # anc_neg_detach = anc_neg_weighted_patches
        # pos_detach = pos_weighted_patches
        # neg_detach = neg_weighted_patches

        # cls_norm = torch.norm(cls_feat_detach,p=2,dim=1)
        # anc_pos_norm = torch.norm(anc_pos_weighted,p=2,dim=1)
        # anc_neg_norm = torch.norm(anc_neg_weighted,p=2,dim=1)
        # pos_norm = torch.norm(pos_detach,p=2,dim=1)
        # neg_norm = torch.norm(neg_detach,p=2,dim=1)

        # anc_pos_norm_ratio = (cls_norm / anc_pos_norm).unsqueeze(-1)
        # anc_neg_norm_ratio = (cls_norm / anc_neg_norm).unsqueeze(-1)
        # pos_norm_ratio = (cls_norm[ind_pos_cls] / pos_norm).unsqueeze(-1)
        # neg_norm_ratio = (cls_norm[ind_neg_cls] / neg_norm).unsqueeze(-1)

        anc_diff = (cls_feat - anc_pos_weighted_patches * anc_ratio)
        abs = torch.abs(anc_diff)
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        abs_norm = (abs / (abs_max+1e-12))
        abs_common = 1 - abs_norm

        # anc_pos_weight,anc_neg_weight,pos_weight,neg_weight = 1 - abs_norm[0],abs_norm[1],1 - abs_norm[2],abs_norm[3]
        # abs = torch.cat((anc_pos_weight,anc_neg_weight,pos_weight,neg_weight))
        abs_norm[abs_norm<t] = -self.weight_param
        #abs_common[abs_common<t] = -self.weight_param
        
        abs_norm = abs_norm + self.weight_param
        #abs_common = abs_norm + self.weight_param
        anc_weight = abs_norm

        #abs_diff = abs / (abs_max + 1e-12)
        #abs_common = 1 - abs/ (abs_max + 1e-12)
        #abs_diff[abs_diff < 
        # t] = -self.weight_param 
        #abs_common[abs_common<t] = -self.weight_param
        #abs_diff = abs_diff + self.weight_param
        #abs_common = abs_common + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        #abs_diff = abs_diff.reshape(-1,B,C)
        #abs_common = abs_common.reshape(-1,B,C)
        
        #anc_pos_weight , anc_neg_weight, pos_weight, neg_weight = abs_diff[0],abs_diff[1],abs_diff[2],abs_diff[3]
        #anc_pos_weight_c, anc_neg_weight_c, pos_weight_c, neg_weight_c = abs_common[0],abs_common[1],abs_common[2],abs_common[3]
        
        dist_pos = torch.sum(
            (cls_feat * anc_weight - pos_cls * anc_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_weight - neg_cls * anc_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
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
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            #loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_dist = self.ranking_loss(dist_position_neg - dist_position_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_mean + loss_cls_weighted + JSD_loss
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls
class TripletAttentionLoss_ss_pos_6(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, comb, comb_idx, jsd, head_wise, head_num, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
        self.LW_loss = LayerWise_JSD()
        self.comb = comb 
        self.comb_idx = comb_idx
        self.JSD = jsd
        self.HEAD_WISE = head_wise
        self.HEAD_NUM = head_num
        self.KLD_loss = nn.KLDivLoss(reduction='batchmean')
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
        rel_pos_bias,
        abs_pos,
        cls_param,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
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
        param = cls_param[labels]
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

        ####################################################################################################
        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        pos_sim = (((pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)
        
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)
        
        # Performance Index 242
        # pos_sim = anc_sim[ind_pos_cls]
        # neg_sim = anc_sim[ind_neg_cls]
        
        # rank = int(N* self.patch_ratio)
        # p_ratio = ((self.patch_ratio[1]-self.patch_ratio[0])/(self.max_epoch-1))*(epoch-1) + self.patch_ratio[0]
        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = torch.topk(pos_sim,rank,dim=-1)
        ####################################################################################################

        param_sim= (((param.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        val_anc_param,ind_anc_param = torch.topk(param_sim,self.comb_idx,dim=-1)
        if self.comb :
            anc_comb = [torch.combinations(x,r=2) for x in ind_anc_param]
        else :
            anc_comb = [torch.cartesian_prod(x,x) for x in ind_anc_param]

        anc_comb = torch.cat(anc_comb,0).reshape(B,-1,2)
        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[:,x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                abs_val_anc = abs_val_anc.unsqueeze(1).expand(rel_val_anc.size(0),rel_val_anc.size(1))
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)
        anc_vec = (torch.cat(anc_vec).reshape(B,anc_comb.size(1),-1).transpose(-2,-1).softmax(-1)).reshape(B,-1)
        
        if self.comb :
            if self.HEAD_WISE:
                anc_vec = anc_vec.reshape(B,-1,int((self.comb_idx * (self.comb_idx -1))/2))
            else :
                anc_vec = anc_vec.reshape(B,-1,int((self.HEAD_NUM * self.comb_idx * (self.comb_idx-1))/2))
        else :
            if self.HEAD_WISE:
                anc_vec = anc_vec.reshape(B,-1,self.comb_idx * self.comb_idx)
            else :
                anc_vec = anc_vec.reshape(B,-1,self.HEAD_NUM * self.comb_idx * self.comb_idx)
        pos_vec = anc_vec[ind_pos_cls]
        
        #neg_vec = anc_vec[ind_neg_cls]
        
        #################################################################################
        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]

        anc_pos_val = [anc_sim[i][intersect_pos[i]] for i in range(B)]
        anc_pos_weighted_patches = torch.stack([torch.sum(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)]) # Anc Pos common patch - for Anc
        anc_diff = (cls_feat - anc_pos_weighted_patches * anc_ratio)
        abs = torch.abs(anc_diff)
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        abs_norm = (abs / (abs_max+1e-12))

        # anc_pos_weight,anc_neg_weight,pos_weight,neg_weight = 1 - abs_norm[0],abs_norm[1],1 - abs_norm[2],abs_norm[3]
        # abs = torch.cat((anc_pos_weight,anc_neg_weight,pos_weight,neg_weight))
        abs_norm[abs_norm<t] = -self.weight_param
        #abs_common[abs_common<t] = -self.weight_param
        
        abs_norm = abs_norm + self.weight_param
        #abs_common = abs_norm + self.weight_param
        anc_weight = abs_norm
        ################################################################################

        # anc_pos_val = [anc_sim[i][intersect_pos[i]]/anc_sim[i][intersect_pos[i]].max()for i in range(B)]
        # anc_neg_val = [anc_sim[i][intersect_neg[i]]/anc_sim[i][intersect_neg[i]].max() for i in range(B)]
         
        # pos_val = [pos_sim[i][intersect_pos[i]]/pos_sim[i][intersect_pos[i]].max() for i in range(B)]
        # neg_val = [neg_sim[i][intersect_neg[i]]/neg_sim[i][intersect_neg[i]].max() for i in range(B)]

        
        # cat_all = torch.cat((ind_pos,ind_neg,ind_anc),dim=-1)
        # cat_all_idx, cat_all_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_all],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_all]
        
        # intersect_all = [cat_all_idx[i][cat_all_cnts[i]==3] for i in range(B)]
        # patches_all = [patch_feat_A[i][intersect_all[i]] for i in range(B)]
        # anc_all_val = [anc_sim[i][intersect_all[i]]/anc_sim[i][intersect_all[i]].max() for i in range(B)]

        # cls_feat_detach = cls_feat
        # anc_pos_detach = anc_pos_weighted_patches
        # anc_neg_detach = anc_neg_weighted_patches
        # pos_detach = pos_weighted_patches
        # neg_detach = neg_weighted_patches

        # cls_norm = torch.norm(cls_feat_detach,p=2,dim=1)
        # anc_pos_norm = torch.norm(anc_pos_weighted,p=2,dim=1)
        # anc_neg_norm = torch.norm(anc_neg_weighted,p=2,dim=1)
        # pos_norm = torch.norm(pos_detach,p=2,dim=1)
        # neg_norm = torch.norm(neg_detach,p=2,dim=1)

        # anc_pos_norm_ratio = (cls_norm / anc_pos_norm).unsqueeze(-1)
        # anc_neg_norm_ratio = (cls_norm / anc_neg_norm).unsqueeze(-1)
        # pos_norm_ratio = (cls_norm[ind_pos_cls] / pos_norm).unsqueeze(-1)
        # neg_norm_ratio = (cls_norm[ind_neg_cls] / neg_norm).unsqueeze(-1)



        #abs_diff = abs / (abs_max + 1e-12)
        #abs_common = 1 - abs/ (abs_max + 1e-12)
        #abs_diff[abs_diff < 
        # t] = -self.weight_param 
        #abs_common[abs_common<t] = -self.weight_param
        #abs_diff = abs_diff + self.weight_param
        #abs_common = abs_common + self.weight_param # normalize 후 0.1 보다 작은 값은 0으로
        #abs_diff = abs_diff.reshape(-1,B,C)
        #abs_common = abs_common.reshape(-1,B,C)
        
        #anc_pos_weight , anc_neg_weight, pos_weight, neg_weight = abs_diff[0],abs_diff[1],abs_diff[2],abs_diff[3]
        #anc_pos_weight_c, anc_neg_weight_c, pos_weight_c, neg_weight_c = abs_common[0],abs_common[1],abs_common[2],abs_common[3]
        
        dist_pos = torch.sum(
            (cls_feat * anc_weight - pos_cls * anc_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_weight - neg_cls * anc_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        # dist_ap = torch.gather(dist_mat,1,ind_pos_cls.unsqueeze(-1))
        # dist_an = torch.gather(dist_mat,1,ind_neg_cls.unsqueeze(-1))
        # dist_ap = dist_ap.squeeze(1)
        # dist_an = dist_an.squeeze(1)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            #loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            if self.JSD : 
                if self.HEAD_WISE : 
                    position_loss = self.JSD_loss(anc_vec,pos_vec)
                else :
                    position_loss = self.LW_loss(anc_vec,pos_vec)
            else :
                position_loss = self.KLD_loss(anc_vec.log(),pos_vec.log())
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            # loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            # loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_dist = self.ranking_loss(dist_position_neg - dist_position_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_mean + loss_cls_weighted + position_loss 
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls

class TripletAttentionLoss_ss_pos_7(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, comb, comb_idx,margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.comb = comb
        self.comb_idx = comb_idx
        self.JSD_loss = JSD()
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
        rel_pos_bias,
        abs_pos,
        cls_param,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
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
        param = cls_param[labels]

        anc_sim = (((param.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)  
        # anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        # cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        # anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)
        
        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        
        # dummy_idx = ind_anc.unsqueeze(2).expand(ind_anc.size(0),ind_anc.size(1),patch_feat_A.size(2))
        # rank_patch  = patch_feat_A.gather(1,dummy_idx)
        # anc_main_patches = torch.sum(val_anc.unsqueeze(-1) * rank_patch,dim=1) 

        # diff_cls = (cls_feat - anc_main_patches * anc_ratio)

        ind_anc_pos = ind_anc[:,:self.comb_idx]

        if self.comb :
            anc_comb = [torch.combinations(x,r=2) for x in ind_anc_pos]
        else : 
            anc_comb = [torch.cartesian_prod(x,x) for x in ind_anc_pos]
        anc_comb = torch.cat(anc_comb,0).reshape(B,-1,2)
        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[:,x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                abs_val_anc = abs_val_anc.unsqueeze(1).expand(rel_val_anc.size(0),rel_val_anc.size(1))
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)
        anc_vec = (torch.cat(anc_vec).reshape(B,anc_comb.size(1),-1).transpose(-2,-1).softmax(-1)).reshape(B,-1)
        pos_vec = anc_vec[ind_pos_cls]

        JSD_loss = self.JSD_loss(anc_vec,pos_vec)

        # dist_pos = torch.sum(
        #     (diff_cls - diff_cls[ind_pos_cls]).pow(2), dim=1
        # ).sqrt()
        
        # dist_neg = torch.sum(
        #     (diff_cls - diff_cls[ind_neg_cls]).pow(2), dim=1
        # ).sqrt() # * : element wise multiplication
        
        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            #loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            #loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_dist = self.ranking_loss(dist_position_neg - dist_position_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls + JSD_loss
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls

class TripletAttentionLoss_ss_pos_3(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
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
        rel_pos_bias,
        abs_pos,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
 
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
        # Method 5

        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 

        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,10,dim=-1)
        anc_comb = [torch.combinations(x,r=2) for x in ind_anc]

        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)

        anc_vec = (torch.cat(anc_vec).reshape(B,-1)).softmax(-1)
        pos_vec = anc_vec[ind_pos_cls]
        #neg_vec = anc_vec[ind_neg_cls]
        
        JSD_loss = self.JSD_loss(anc_vec,pos_vec)
        
        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)

        else:
            loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            loss = loss_cls + JSD_loss
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls

class TripletAttentionLoss_ss_pos_4(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
        self.JSD_loss = JSD()
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
        rel_pos_bias,
        abs_pos,
        cls_param,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        #global_feat = global_feat.contiguous()
        t = 0
        cls_feat = global_feat[:,0] # detach()
        dist_mat_cls = euclidean_dist(cls_feat,cls_feat)
        params = cls_param[labels]
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
        # Method 5

        anc_sim = (((params.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 

        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,10,dim=-1)
        anc_comb = [torch.combinations(x,r=2) for x in ind_anc]

        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)

        anc_vec = (torch.cat(anc_vec).reshape(B,-1)).softmax(-1)
        pos_vec = anc_vec[ind_pos_cls]
        #neg_vec = anc_vec[ind_neg_cls]
        
        JSD_loss = self.JSD_loss(anc_vec,pos_vec)
        
        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)

        else:
            loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            loss = loss_cls + JSD_loss
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls

class TripletAttentionLoss_ss_pos_1(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, loss_ratio, patch_ratio, num_instance, max_epoch, rel_pos, margin: Optional[float] = None, hard_factor=0.0):
        self.margin = margin
        self.attn_loss = nn.MSELoss()
        self.hard_factor = hard_factor
        self.patch_ratio = patch_ratio
        self.loss_ratio = loss_ratio
        self.num_instance = num_instance
        self.max_epoch = max_epoch
        self.rel_pos = rel_pos
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
        rel_pos_bias,
        abs_pos,
        normalize_feature: bool = False,
    ) -> Tuple[torch.Tensor]:
        t = 0

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
        
        # Method 5
        pos_cls = cls_feat[ind_pos_cls]
        neg_cls = cls_feat[ind_neg_cls]

        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        pos_sim = (((pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)
        
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)
        
        p_ratio = self.patch_ratio[0]
        rank = int(N*p_ratio)
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = torch.topk(pos_sim,rank,dim=-1)
        
        ind_anc_5 = ind_anc[:,:5]

        anc_comb = [torch.combinations(x,r=2) for x in ind_anc_5]

        anc_vec = []
        for i in range(B):
            if self.rel_pos :
                rel_val_anc = torch.stack([rel_pos_bias[x[0],x[1]] for x in anc_comb[i]])
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(rel_val_anc+abs_val_anc)
            else :
                abs_val_anc = torch.stack([abs_pos[x[0]] @ abs_pos[x[1]] for x in anc_comb[i]])
                anc_vec.append(abs_val_anc)

        anc_vec = torch.cat(anc_vec).reshape(B,-1)
        pos_vec = anc_vec[ind_pos_cls]
        neg_vec = anc_vec[ind_neg_cls]
        
        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]

        anc_pos_val = [anc_sim[i][intersect_pos[i]] for i in range(B)]

        anc_pos_weighted_patches = torch.stack([torch.sum(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)]) # Anc Pos common patch - for Anc

        anc_diff = (cls_feat - anc_pos_weighted_patches * anc_ratio)
        abs = torch.abs(anc_diff)
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        abs_norm = (abs / (abs_max+1e-12))
        abs_common = 1 - abs_norm

        abs_norm[abs_norm<t] = -self.weight_param
        #abs_common[abs_common<t] = -self.weight_param
        
        abs_norm = abs_norm + self.weight_param
        #abs_common = abs_norm + self.weight_param
        anc_weight = abs_norm
        anc_common_weight = abs_common

        dist_pos = torch.sum(
            (cls_feat * anc_weight - pos_cls * anc_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_weight - neg_cls * anc_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
        dist_position_pos = torch.sum(
            (anc_vec - pos_vec).pow(2),dim=1
        ).sqrt()

        dist_position_neg = torch.sum(
            (anc_vec - neg_vec).pow(2),dim=1
        ).sqrt()

        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)


        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            # loss_gap = self.ranking_loss(dist_an, dist_ap, y)
            # loss = loss_cls + 0.2 * loss_gap
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_gap = self.ranking_loss(dist_an - dist_ap, y)
            #loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            loss_dist = self.ranking_loss(dist_position_neg - dist_position_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_mean + loss_cls_weighted + loss_dist
            
            if torch.isnan(loss) or torch.isinf(loss) :
                wandb.finish()

        return loss, p_ratio, dist_ap_cls, dist_an_cls

class TripletAttentionLoss_ss_1(object):
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
        t = 0
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

        pos_cls = cls_feat[ind_pos_cls]
        neg_cls = cls_feat[ind_neg_cls]

        anc_sim = (((cls_feat.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        pos_sim = (((pos_cls.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1)
        
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)

        p_ratio = ((self.patch_ratio[1]-self.patch_ratio[0])/(self.max_epoch-1))*(epoch-1) + self.patch_ratio[0]
        rank = int(N*p_ratio)
        
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        val_pos, ind_pos = torch.topk(pos_sim,rank,dim=-1)

        cat_pos = torch.cat((ind_anc,ind_pos),dim=-1)

        cat_pos_idx, cat_pos_cnts = [torch.unique(x,return_counts=True,dim=0)[0] for x in cat_pos],[torch.unique(x,return_counts=True,dim=0)[1] for x in cat_pos]
        intersect_pos = [cat_pos_idx[i][cat_pos_cnts[i]!=1] for i in range(B)]
        patches_pos = [patch_feat_A[i][intersect_pos[i]] for i in range(B)]
         
        anc_pos_val = [anc_sim[i][intersect_pos[i]] for i in range(B)]

        anc_pos_weighted_patches = torch.stack([torch.sum(patches_pos[i]*anc_pos_val[i].unsqueeze(-1),dim=0) for i in range(B)]) # Anc Pos common patch - for Anc

        anc_diff = (cls_feat - anc_pos_weighted_patches * anc_ratio)
        abs = torch.abs(anc_diff)
        abs_max , _ = torch.max(abs,dim=1,keepdim=True)
        abs_norm = (abs / (abs_max+1e-12))
        # abs_common = 1 - abs_norm

        abs_norm[abs_norm<t] = -self.weight_param
        #abs_common[abs_common<t] = -self.weight_param
        
        abs_norm = abs_norm + self.weight_param
        # abs_common = abs_norm + self.weight_param
        anc_weight = abs_norm
        # anc_common_weight = abs_common

        dist_pos = torch.sum(
            (cls_feat * anc_weight - pos_cls * anc_weight).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (cls_feat * anc_weight - neg_cls * anc_weight).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
        # dist_pos_common = torch.sum(
        #    (cls_feat * anc_common_weight - pos_cls * anc_common_weight).pow(2), dim=1
        # ).sqrt()

        # dist_neg_common = torch.sum(
        #     (cls_feat * anc_common_weight - neg_cls * anc_common_weight).pow(2), dim=1
        # ).sqrt()

        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_weighted + loss_cls_mean
        return loss, p_ratio, dist_ap_cls, dist_an_cls
class TripletAttentionLoss_ss_2(object):
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
        t = 0
        cls_feat = global_feat[:,0] # detach()
        dist_mat_cls = euclidean_dist(cls_feat,cls_feat)
        
        (
            dist_ap_cls,
            dist_an_cls,
            dist_an_mean_cls,
            ind_pos_cls,
            ind_neg_cls,
        ) = hard_example_mining_with_inds(dist_mat_cls, labels)
        
        cls_param = cls_param.detach()
        anc_param = cls_param[labels]
        patch_feat_A = global_feat[:,1:]
        B,N,C = patch_feat_A.shape
        ID = self.num_instance
        scale = cls_feat.shape[-1] ** 0.5

        anc_sim = (((anc_param.unsqueeze(1) @ patch_feat_A.transpose(-1,-2)).squeeze(1))/scale).softmax(-1) 
        anc_patch_norm = torch.norm(torch.sum(anc_sim.unsqueeze(-1)*patch_feat_A,dim=1),p=2,dim=1)
        cls_feat_norm = torch.norm(cls_feat,p=2,dim=1)
        
        anc_ratio = (cls_feat_norm / anc_patch_norm).unsqueeze(-1)

        p_ratio = ((self.patch_ratio[1]-self.patch_ratio[0])/(self.max_epoch-1))*(epoch-1) + self.patch_ratio[0]
        rank = int(N*p_ratio)
        
        val_anc, ind_anc = torch.topk(anc_sim,rank,dim=-1)
        
        dummy_idx = ind_anc.unsqueeze(2).expand(ind_anc.size(0),ind_anc.size(1),patch_feat_A.size(2))
        rank_patch  = patch_feat_A.gather(1,dummy_idx)
        anc_main_patches = torch.sum(val_anc.unsqueeze(-1) * rank_patch,dim=1) 

        diff_cls = (cls_feat - anc_main_patches * anc_ratio)

        dist_pos = torch.sum(
            (diff_cls - diff_cls[ind_pos_cls]).pow(2), dim=1
        ).sqrt()
        
        dist_neg = torch.sum(
            (diff_cls - diff_cls[ind_neg_cls]).pow(2), dim=1
        ).sqrt() # * : element wise multiplication
        
        # dist_pos_common = torch.sum(
        #    (cls_feat * anc_common_weight - pos_cls * anc_common_weight).pow(2), dim=1
        # ).sqrt()

        # dist_neg_common = torch.sum(
        #     (cls_feat * anc_common_weight - neg_cls * anc_common_weight).pow(2), dim=1
        # ).sqrt()

        dist_ap_cls *= (1.0 + self.hard_factor)
        dist_an_cls *= (1.0 + self.hard_factor)

        y = dist_an_cls.new().resize_as_(dist_an_cls).fill_(1)
        if self.margin is not None:
            loss_cls = self.ranking_loss(dist_an_cls, dist_ap_cls, y)
            loss_cls_weighted = self.ranking_loss(dist_neg, dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common,dist_pos_common,y)
            #loss_cls_mean = self.ranking_loss(dist_an_mean_cls, dist_ap_cls,y)
            #loss =  loss_cls_weighted_common + loss_cls_weighted + loss_cls
        else:
            #loss_cls = self.ranking_loss(dist_an_cls - dist_ap_cls, y)
            #loss_cls_detach = self.ranking_loss(dist_an_cls - dist_ap_cls.detach(),y)
            loss_cls_mean = self.ranking_loss(dist_an_mean_cls - dist_ap_cls.detach(),y)
            loss_cls_weighted = self.ranking_loss(dist_neg - dist_pos,y)
            #loss_cls_weighted_common = self.ranking_loss(dist_neg_common - dist_pos_common,y)
            #loss =  (1-self.loss_ratio) *loss_cls_weighted_common + self.loss_ratio * loss_cls_weighted
            loss = loss_cls_weighted + loss_cls_mean
        if torch.isnan(loss) or torch.isinf(loss) :
            wandb.finish()
        # elif epoch == 41 :
        #     wandb.finish()
        return loss, p_ratio, dist_ap_cls, dist_an_cls