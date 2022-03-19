# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, TripletAttentionLoss
from .center_loss import CenterLoss
import torch
from typing import Tuple

# Loss 는 class로 구성
def make_loss(cfg, num_classes):    # make loss는 class가 아닌 definition
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True) 
    # center loss는 parameter가 존재, nn.Module을 상속받는 class
    # center loss는 classifier단의 weight로 loss를 구하는 것이 아니라, 자체적인 parameter를 optimize하기 때문에 criterion을 따로 구성해야함   
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss() # __call__ return : loss, dist_ap, dist_an
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    elif 'hnewth' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletAttentionLoss()
            print("using element weighted triplet loss for training")
        else :
            triplet = TripletAttentionLoss(cfg.SOLVER.MARGIN)
            print("using element weighted triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet or hnewth''but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': 
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            def loss_func(score, feat, target, target_cam):
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list): 
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target) # LabelSmooth

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]] # Equation (4)
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list): 
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

        elif 'hnewth' in cfg.MODEL.METRIC_LOSS_TYPE:
            def loss_func(score, feat, target, target_cam, cls_param):
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list): 
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target) # LabelSmooth

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target, cls_param)[0] for feats in feat[1:]] # Equation (4)
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target, cls_param)[0]
                    else:
                        TRI_LOSS = triplet(feat, target, cls_param)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list): 
                            TRI_LOSS = [triplet(feats, target, cls_param)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target, cls_param)[0]
                    else:
                            TRI_LOSS, HTH, TH, HNTH_P2  = triplet(feat, target, cls_param)

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, HTH, TH, HNTH_P2

        else:
            print('expected METRIC_LOSS_TYPE should be triplet, hnewth''but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion # return 값이 definition, 즉 함수임


