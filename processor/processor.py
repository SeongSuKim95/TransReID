import logging
import os
from re import T
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import shutil
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, demo
from torch.cuda import amp
import torch.distributed as dist
from .vit_rollout import VITAttentionRollout
import wandb

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    if cfg.WANDB : 
        wandb.init(project="TransReID", entity="panda0728",config=cfg)
        #wandb.watch(model,loss_fn, log = "all", log_freq = 1)
        cfg_wb = wandb.config
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    triplet_type = cfg.MODEL.METRIC_LOSS_TYPE
    REL_POS = cfg.MODEL.REL_POS
    ABS_POS = cfg.MODEL.ABS_POS
    NUM_HEADS = cfg.MODEL.HEAD_NUM
    num_layers = 12
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN: # 여러개의 GPU를 동시에 사용할떄 
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter() # Averaging acc, loss 
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM) # Evaluating R1, mAP
    scaler = amp.GradScaler()
    if "pos" in triplet_type:
        tri_loss_meter = AverageMeter()
        pp_loss_meter = AverageMeter()
        
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train() 
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad() # center loss를 위한 optimizer
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                # cls score는 bnneck을 통과한 이후의 feature가 classification layer를 통과하여 얻음, 이를 이용하여 ID loss 계산
                # 반면 triplet loss의 경우 base model만을 통과한 global_feature를 이용해서 계산                

                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                # JPM을 사용할 경우
                # score, feat의 개수는 JPM branch 개수와 같다
                # score.size = [#JPM,bs,train_ID]  [5,64,751]
                # feat.size  = [#JPM,bs,feat_size] [5,64,768]
                if 'pos' in triplet_type:
                    if REL_POS :
                        bias_index = model.base.blocks[-1].attn.state_dict()['relative_position_index']
                        patch_num = bias_index.size(0)
                        bias_index = bias_index.view(-1)
                        table_list = []
                        for i in range(num_layers):
                            table_list.append(model.base.blocks[i].attn.state_dict()['relative_position_bias_table'])
                        bias_table = torch.cat(table_list,1).T
                        bias_index_dummy = bias_index.unsqueeze(0).expand(bias_table.size(0),bias_index.size(0))
                        rel_pos_bias = bias_table.gather(1,bias_index_dummy).reshape(-1,patch_num,patch_num)
                    if ABS_POS :
                        abs_pos = model.base.pos_embed[0]
                    loss, tri_loss, pp_loss = loss_fn(score,feat,target,target_cam,rel_pos_bias,abs_pos,model.classifier.state_dict()["weight"])
                else : 
                    loss = loss_fn(score,feat,target,target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
                # center loss의 parameter에 대해서도 update
            if isinstance(score, list): # JPM 
                acc = (score[0].max(1)[1] == target).float().mean() # Train_accuracy는 global_branch score로 결정
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            if "pos" in triplet_type:
                tri_loss_meter.update(tri_loss.item(),img.shape[0])   
                pp_loss_meter.update(pp_loss.item(),img.shape[0])
            torch.cuda.synchronize() # cuda의 work group 내의 모든 wavefront속 kernel이 전부 연산을 마칠때까지 기다려줌 
            if (n_iter + 1) % log_period == 0:

                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                if cfg.WANDB : 
                    if triplet_type == 'triplet_ss_pos_6':
                        wandb.log({ 'Train Epoch': epoch, 
                                    'loss' : loss_meter.avg, 
                                    'Learning rate': scheduler._get_lr(epoch)[0],
                                    'tri_loss': tri_loss_meter.avg,
                                    'pp_loss' : pp_loss_meter.avg,
                                    'Acc': acc_meter.avg})                      
                    else :
                         wandb.log({'Train Epoch': epoch, 
                                    'loss' : loss_meter.avg, 
                                    'Learning rate': scheduler._get_lr(epoch)[0]})
        end_time = time.time() #Epoch마다 걸리는 시간 측정      
        time_per_batch = (end_time - start_time) / (n_iter + 1) # Batch수로 나누어주어 batch마다 걸리는 시간 측정
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, str(cfg.INDEX), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, str(cfg.INDEX), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _, _, _, _, _, _  = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if cfg.WANDB : 
                    wandb.log({'Val Epoch': epoch, 'mAP' : mAP, 'Rank1' : cmc[0], 'Rank5': cmc[4], 'Rank10': cmc[9]})
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 args,
                 q_dir,
                 g_dir,
                 ):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        print(n_iter)
        with torch.no_grad():
            img = img.to(device) # [256, 3, 256, 256] (Batch 256)
            camids = camids.to(device) # [256]
            target_view = target_view.to(device)
            # if cfg.TEST.VISUALIZE :
            #     attention_rollout =  VITAttentionRollout(model,head_fusion=cfg.TEST.HEAD_FUSION, discard_ratio=cfg.TEST.DISCARD_RATIO)
            #     feat, mask = attention_rollout(img)
            #     if not n_iter :
            #         mask_list = mask
            #     else :
            #         mask_list = np.concatenate((mask_list,mask))
            # else :
            # feat = model(img, cam_label=camids, view_label=target_view) # [256,768]
            
            feat = model(img, cam_label=camids, view_label=target_view) # [256,768]
            
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
    # feats : [76, 256, 768]
    cmc, mAP, distmat_eucd, distmat_cos, pids_all, camids_all, qf, gf, q_pids, g_pids, q_camids, g_camids = evaluator.compute() # cmc[i] = Rank i score
    # distmat : (num_query,num_gallery) compared by euclidean distance
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    result = {'gallery_f':gf.numpy(),'gallery_label':g_pids,'gallery_cam':g_camids,'query_f':qf.numpy(),'query_label':q_pids,'query_cam':q_camids,'img_path': img_path_list,'q_dir':q_dir,'g_dir':g_dir,'Euclidean_dist':distmat_eucd, 'Cos_dist':distmat_cos} # type(label) ,type(cam) = list , type(feature)= torch.tensor
    path = 'result/result_matrix'
    os.makedirs(path,exist_ok=True)
    scipy.io.savemat(f'{path}/{cfg.INDEX}.mat',result)    
    if cfg.TEST.VISUALIZE :
        os.system(f'python -m processor.demo --config_file={args.config_file}')
    return cmc[0], cmc[4] # Rank 1, Rank 5


