import torch
import numpy as np
import os
from utils.reranking import re_ranking
from config import cfg

def demo(qf,gf):
    return

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g)) # gallery가 rank 수보다 작을 경우
    indices = np.argsort(distmat, axis=1) # indices.shape = (3368,15913) 각 query에 대해 gallery마다 순위를 index로 매김
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32) # 거리순으로 정렬된 gallery들의 pid가 query의 pid와 같은지 확인
    # q_pids         q_pids[ : , np.newaxis]
    # [1,2,3,4]         [[1],[2],[3],[4]]

    # matches.shape = (3368,15913)
    # match 된경우 1, 그렇지 않은 경우 0
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row, q_idx번째 query의 gallery distance 순위
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid) # 같은 cam_id, 같은 pid를 가지는 gallery sample은 제외
        keep = np.invert(remove)

        # compute cmc curve
        # cmc = cumulative matching characteristics 
        # For each query, an algorithm will rank all the gallery samples according to their distances to the query from small to large, and the CMC top-k accuracy is
        # acc_k = 1 (if top-k ranked gallery samples contain the query identity)
        #       = 0 (otherwise)
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep] 
        # matches : 같은 pid 갖는지 , keep : (같은 pid,같은 cam_id)가 아닌 id
        # a = np.array([1,1,1,1]), b = np.array([False,True,False,True])
        # a[b] = [1,1]
        # 즉, 이렇게 함으로써 query와 같은 pid를 가지면서 다른 cam_id를 갖는 gallery만을 골라낸다
        
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        # np.cumsum()
        # a = [[1,2,3],[4,5,6]]
        # np.cumsum(a) = [1,3,6,10,15,21]
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum() # gallery 개수
        tmp_cmc = orig_cmc.cumsum() # gallery cumulative sum
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0 # y = [1,2,3,4...]
        tmp_cmc = tmp_cmc / y  
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    # len(all_AP) : query.size (3368)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP) # 모든 query의 Average precision의 평균

    return all_cmc, mAP

class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu()) #[batch_size, 768]
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0) # [num_batch, batch_size, feat_dim]
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # feats.size() = [#query + #gallery, feat_dim] : Val_loader에서 함께 들어왔으므로!
        # self.num_query = 3368
        qf = feats[:self.num_query] # query_feature
        q_pids = np.asarray(self.pids[:self.num_query]) # query pid 
        q_camids = np.asarray(self.camids[:self.num_query]) # query cam id 
        # gallery -> self.num_query to end  
        gf = feats[self.num_query:] # gallery feature
        g_pids = np.asarray(self.pids[self.num_query:]) # gallery pid
        g_camids = np.asarray(self.camids[self.num_query:]) # gallery cam id
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            distmat_eucd = euclidean_distance(qf, gf) # qf.size() = [3368,768], , gf.size() = [15913,768]
            distmat_cos = cosine_similarity(qf, gf)
        if cfg.TEST.EVAL_METRIC == 'Euclidean':
            print('=> Computing DistMat with euclidean_distance')
            cmc, mAP = eval_func(distmat_eucd, q_pids, g_pids, q_camids, g_camids) 
        elif cfg.TEST.EVAL_METRIC == 'Cos':
            print('=> Computing DistMat with cos_similarity')
            cmc, mAP = eval_func(distmat_cos, q_pids, g_pids, q_camids, g_camids) 
        else :
            raise NotImplementedError("Visualize metric should be Euclidean or Cosine similarity")

        # dist_mat.size() = [3368,15913], len(q_pids) = 3368, len(g_pids) = 15913, len(q_camids) = 3368, len(g_camids) = 15913
        return cmc, mAP, distmat_eucd, distmat_cos, self.pids, self.camids, qf, gf, q_pids, g_pids, q_camids, g_camids



