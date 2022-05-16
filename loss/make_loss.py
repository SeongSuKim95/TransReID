# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import  TripletLoss, TripletAttentionLoss_ss_1, TripletAttentionLoss_ss_2, TripletAttentionLoss_ss_pos_1, TripletAttentionLoss_ss_pos_2,TripletAttentionLoss_ss_pos_3, TripletAttentionLoss_ss_pos_4,TripletAttentionLoss_ss_pos_5,TripletAttentionLoss_ss_pos_6,TripletAttentionLoss_ss_pos_7
from .center_loss import CenterLoss
import torch
from typing import Tuple

# Loss 는 class로 구성
def make_loss(cfg, num_classes):    # make loss는 class가 아닌 definition
    sampler = cfg.DATALOADER.SAMPLER
    loss_type = cfg.MODEL.METRIC_LOSS_TYPE
    patch_ratio = cfg.SOLVER.PATCH_RATIO
    loss_ratio = cfg.SOLVER.LOSS_RATIO
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    max_epoch = cfg.SOLVER.MAX_EPOCHS
    feat_norm = cfg.SOLVER.FEAT_NORM
    rel_pos = cfg.MODEL.REL_POS
    comb = cfg.SOLVER.COMB
    comb_idx = cfg.SOLVER.COMB_INDEX
    jsd = cfg.SOLVER.JSD
    head_wise = cfg.SOLVER.HEAD_WISE
    head_num = cfg.MODEL.HEAD_NUM
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True) 
    # center loss는 parameter가 존재, nn.Module을 상속받는 class
    # center loss는 classifier단의 weight로 loss를 구하는 것이 아니라, 자체적인 parameter를 optimize하기 때문에 criterion을 따로 구성해야함   
    if "triplet" in sampler :
        if loss_type == "triplet":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletLoss(feat_norm) # __call__ return : loss, dist_ap, dist_an
                print("using soft triplet loss for training")
            else:
                triplet = TripletLoss(feat_norm,cfg.SOLVER.MARGIN)  # triplet loss
                print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
        elif loss_type == "triplet_ss_1":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_1(loss_ratio,patch_ratio,num_instance,max_epoch)
                print("using soft triplet_ss_1 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_1(loss_ratio,patch_ratio,num_instance,max_epoch,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_1 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))
        elif loss_type == "triplet_ss_2":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_2(loss_ratio,patch_ratio,num_instance,max_epoch)
                print("using soft triplet_ss_2 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_2(loss_ratio,patch_ratio,num_instance,max_epoch,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_2 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))        
        elif loss_type == "triplet_ss_pos_1":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_1(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos)
                print("using soft triplet_pos_1 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_1(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_pos_1 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))        
        elif loss_type == "triplet_ss_pos_2":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_2(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos)
                print("using soft triplet_ss_pos_2 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_2(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_2 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))                
        elif loss_type == "triplet_ss_pos_3":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_3(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos)
                print("using soft triplet_ss_pos_3 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_3(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_3 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))         
        elif loss_type == "triplet_ss_pos_4":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_4(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos)
                print("using soft triplet_ss_pos_4 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_4(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_4 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN))                       
        elif loss_type == "triplet_ss_pos_5":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_5(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos)
                print("using soft triplet_ss_pos_5 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_5(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_5 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN)) 
        elif loss_type == "triplet_ss_pos_6":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_6(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,comb,comb_idx,jsd,head_wise,head_num)
                print("using soft triplet_ss_pos_6 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_6(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,comb,comb_idx,jsd,head_wise,head_num,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_6 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN)) 
        elif loss_type == "triplet_ss_pos_7":
            if cfg.MODEL.NO_MARGIN:
                triplet = TripletAttentionLoss_ss_pos_7(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,comb,comb_idx)
                print("using soft triplet_ss_pos_7 attention loss for training with loss ratio : {} ,patch ratio : {}".format(loss_ratio,patch_ratio))
            else:
                triplet = TripletAttentionLoss_ss_pos_7(loss_ratio,patch_ratio,num_instance,max_epoch,rel_pos,comb,comb_idx,cfg.SOLVER.MARGIN)  # triplet loss
                print("using soft triplet_ss_pos_7 attention loss with loss_ratio : {}, patch ratio : {}, margin:{}".format(loss_ratio,patch_ratio,cfg.SOLVER.MARGIN)) 
 
        else:
            print('expected METRIC_LOSS_TYPE should be triplet/triplet_ss/triplet_ss_1/triplet_ss_2''but got {}'.format(loss_type))
        
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax': 
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        if loss_type == 'triplet':
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
        
        elif loss_type in ['triplet_ss_1','triplet_ss_2']:
            def loss_func(score, feat,target,target_cam,epoch,cls_param):
            #def loss_func(score, feat,target,target_cam,epoch,rel_pos_bias,abs_pos):
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
                        TRI_LOSS,PATCH_RATIO = triplet(feat, target, epoch, cls_param)[0] , triplet(feat, target, epoch, cls_param)[1]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, PATCH_RATIO

        elif loss_type in ['triplet_ss_pos_1','triplet_ss_pos_2','triplet_ss_pos_3','triplet_ss_pos_5']:
            #def loss_func(score, feat,target,target_cam,epoch,cls_param,pos_param):
            def loss_func(score, feat,target,target_cam,epoch,rel_pos_bias,abs_pos):
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
                        #TRI_LOSS,PATCH_RATIO = triplet(feat, target, epoch, cls_param, pos_param)[0] , triplet(feat, target, epoch, cls_param, pos_param)[1]
                         TRI_LOSS,PATCH_RATIO = triplet(feat, target, epoch, rel_pos_bias, abs_pos)[0] , triplet(feat, target, epoch, rel_pos_bias, abs_pos)[1]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, PATCH_RATIO
        elif loss_type in ['triplet_ss_pos_4','triplet_ss_pos_6','triplet_ss_pos_7','triplet_ss_pos_8']:
            #def loss_func(score, feat,target,target_cam,epoch,cls_param,pos_param):
            def loss_func(score, feat,target,target_cam,epoch,rel_pos_bias,abs_pos,cls_param):
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
                        #TRI_LOSS,PATCH_RATIO = triplet(feat, target, epoch, cls_param, pos_param)[0] , triplet(feat, target, epoch, cls_param, pos_param)[1]
                        PATCH_RATIO = patch_ratio
                        TRI_LOSS = triplet(feat, target, epoch, rel_pos_bias, abs_pos,cls_param)[0]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, PATCH_RATIO
        else:
            print('expected METRIC_LOSS_TYPE should be triplet, triplet_ss, triplet_ss_1, triplet_ss_2''but got {}'.format(loss_type))
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion # return 값이 definition, 즉 함수임


