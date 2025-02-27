import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID,vit_base_patch16_224_TransReID_SSL
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):
    # features = [bs, #patch_num + 1, feat_dim]
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1) # patch를 shift 수 만큼 shift, 이 떄 cls token은 제외
    # features : 0,1,2,3,4,..,128
    # feature_random : 5,6,7,...,128,1,2,3,4
    x = feature_random # [bs, #patch, dim] 
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim) # [bs, group, #patch/group, dim] shuffle된 patch를 group개로 분할
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous() 
    x = x.view(batchsize, -1, dim)
    
    # a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24],[25,26,27]])
    # b = torch.cat([a[4:], a[1:4]], dim=0)
    # tensor([[13, 14, 15],[16, 17, 18],[19, 20, 21],[22, 23, 24],[25, 26, 27],[ 4,  5,  6],[ 7,  8,  9],[10, 11, 12]])
    # c = b.view(1,4,-1,3)
    # tensor([[[[13, 14, 15],
    #           [16, 17, 18]],
    #          [[19, 20, 21],
    #           [22, 23, 24]],
    #          [[25, 26, 27],
    #           [ 4,  5,  6]],
    #          [[ 7,  8,  9],
    #           [10, 11, 12]]]])
    # d = torch.transpose(c,1,2).contiguous()
    # tensor([[[[13, 14, 15],
    #           [19, 20, 21],
    #           [25, 26, 27],
    #           [ 7,  8,  9]],
    #          [[16, 17, 18],
    #           [22, 23, 24],
    #           [ 4,  5,  6],
    #           [10, 11, 12]]]])
    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module): # nn.Module 상속
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):  # factory : __factory_T_type
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.explain = cfg.TEST.EXPLAIN
        self.IF_CAT = cfg.MODEL.IF_FEAT_CAT
        self.FEAT_NORM = cfg.SOLVER.FEAT_NORM
        self.in_planes = 768
        self.cat_in_planes = 768 * 2
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
        if "SSL" in cfg.MODEL.TRANSFORMER_TYPE:
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                            sie_xishu=cfg.MODEL.SIE_COE,
                                                            camera=camera_num,
                                                            view=view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate= cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                            loss_type = cfg.MODEL.METRIC_LOSS_TYPE,
                                                            gem_pool = cfg.MODEL.GEM_POOLING,
                                                            stem_conv = cfg.MODEL.STEM_CONV,
                                                            rel_pos = cfg.MODEL.REL_POS,
                                                            abs_pos = cfg.MODEL.ABS_POS,
                                                            num_head = cfg.MODEL.HEAD_NUM)
        else :
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                            sie_xishu=cfg.MODEL.SIE_COE,
                                                            camera=camera_num,
                                                            view=view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate= cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                            loss_type = cfg.MODEL.METRIC_LOSS_TYPE,
                                                            ml = cfg.MODEL.ML,
                                                            feat_cat = cfg.MODEL.IF_FEAT_CAT)                                            
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        
        if pretrain_choice == 'imagenet':
            if "SSL" in cfg.MODEL.TRANSFORMER_TYPE:
               self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            else :
               self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
            # Base model 이외에 필요한 layer? : gap, classifier, bnneck layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.METRIC_LOSS_TYPE = cfg.MODEL.METRIC_LOSS_TYPE
        # Loss_type : arcface, cosface, amsoftmax, circleloss 등의 metric_learning은 classifier의 weight를 통해서 구현됨
        # nn.Module을 상속받는 class로 구현
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.IF_CAT :
                self.classifier = nn.Linear(self.cat_in_planes, self.num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)
            else :
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes) #bottle neck layer는 BN
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_patch = nn.BatchNorm1d(self.in_planes) #bottle neck layer는 BN
        self.bottleneck_patch.bias.requires_grad_(False)
        self.bottleneck_patch.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        if global_feat.dim() == 3 : # IF model outputs whole tokens
            cls_feat = global_feat[:,0]
            patch_feat = global_feat[:,1:]  
        else : 
            cls_feat = global_feat
        if self.training:
            if self.IF_CAT : 
                if self.ID_LOSS_TYPE == 'softmax' and self.METRIC_LOSS_TYPE =='':
                    cls_feat = self.bottleneck(cls_feat) # base model을 통과한 feature를 bottleneck layer에 통과   
                    #patch_feat = self.bottleneck_patch(torch.mean(patch_feat,dim=1))
                    feat = torch.cat((cls_feat,torch.mean(patch_feat,dim=1)),dim=1)
                    cls_score = self.classifier(feat)
                elif self.ID_LOSS_TYPE == 'softmax' and self.METRIC_LOSS_TYPE == 'triplet_ss':
                    cls_feat = self.bottleneck(cls_feat)
                    return cls_feat, patch_feat
            else : 
                if self.ID_LOSS_TYPE == 'softmax' and self.METRIC_LOSS_TYPE =='':
                    feat = self.bottleneck(global_feat) # base model을 통과한 feature를 bottleneck layer에 통과
                    cls_score = self.classifier(feat) 
                elif self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    feat = self.bottleneck(global_feat) # base model을 통과한 feature를 bottleneck layer에 통과
                    cls_score = self.classifier(feat, label) # classifier에 label과 함께 통과
                else:
                    if 'pos' in self.METRIC_LOSS_TYPE:
                        bnn_cls_feat = self.bottleneck(cls_feat)
                        cls_score = self.classifier(bnn_cls_feat)
                        return cls_score, global_feat
                    else :
                        feat = self.bottleneck(global_feat) # base model을 통과한 feature를 bottleneck layer에 통과
                        cls_score = self.classifier(feat)
                        return cls_score, global_feat
            # cls score는 bnneck을 통과한 이후의 feature가 classification layer를 통과하여 얻음, 이를 이용하여 ID loss 계산
            # 반면 triplet loss의 경우 base model만을 통과한 global_feature를 이용해서 계산
            if self.FEAT_NORM : 
                return cls_score, feat
            else:
                return cls_score, global_feat  # global feature for triplet loss
        #Inference
        else: # evaluation일때는 feature만 사용 (feature의 bnneck 통과여부는 선택 가능)
            if not self.explain:
                if self.neck_feat == 'after':
                    feat = self.bottleneck(global_feat) # base model을 통과한 feature를 bottleneck layer에 통과
                    # print("Test with feature after BN")
                    return feat
                else:
                    # print("Test with feature before BN")
                    if self.IF_CAT :
                         cls_feat = self.bottleneck(cls_feat)
                         patch_feat = self.bottleneck_patch(torch.mean(patch_feat,dim=1))
                         return torch.cat((cls_feat,patch_feat),dim=1)
                    else :
                        return self.bottleneck(cls_feat)
            else:
                return torch.cat((self.bottleneck(cls_feat),global_feat.squeeze(0)[1:]))
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM,
                                                        camera=camera_num,
                                                        view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        loss_type = cfg.MODEL.METRIC_LOSS_TYPE,
                                                        ml = cfg.MODEL.ML)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1] # Last encoder layer
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label) # features.shape = [bs, #patch + 1, feat_dim]

        # global branch
        b1_feat = self.b1(features) # self.b1 = last layer + layer_norm
        global_feat = b1_feat[:, 0] # cls_token feature, [bs, feat_dim]

        # JPM branch
        feature_length = features.size(1) - 1 # number of patches
        patch_length = feature_length // self.divide_length # divide patches into 4(branch_num) groups
        token = features[:, 0:1] # last layer를 통과하지 않은 features 들중 첫번째 token [bs,1,feat_dim]

        if self.rearrange: 
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups) # features.shape = [bs, #patch + 1, feat_dim], self.shift_num = 5, self.shuffle_groups = 2
        else:
            x = features[:, 1:]
        # lf_1 (local_feature)
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1)) # self.b2 : global_branch와 별도의 encoder layer + layer_norm
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2 (local_feature)
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3 (local_feature)
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4 (local_feature)
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4) 

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else: # Inference 시에는 global_feature랑 local feature들을 concat한 feature 사용
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID_SSL': vit_base_patch16_224_TransReID_SSL,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else: 
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model

