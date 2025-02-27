from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
import wandb 
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/MSMT17/vit_transreid_384_ics_lup_MSMT_Pos.yml", help="path to config file", type=str
    )
    # config_file 이 있다면 defaults configuration에 over ride
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # command line에서도 argument 받아서 over ride
    parser.add_argument("--local_rank", default=0, type=int)
    
    # parser.add_argument("--BASE_LR",default=0, type=float)
    # parser.add_argument("--LOSS_RATIO",default=0, type=float)
    
    parser.add_argument("--COMB_INDEX",default=0, type=int)

    args = parser.parse_args()

    # args.opts.append("SOLVER.BASE_LR")
    # args.opts.append(args.BASE_LR)
    # args.opts.append("SOLVER.LOSS_RATIO")
    # args.opts.append(args.LOSS_RATIO)
    
    args.opts.append("SOLVER.COMB_INDEX")
    args.opts.append(args.COMB_INDEX)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR + "/" + str(cfg.INDEX)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, cfg.INDEX,if_train=True) # Setting loger
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str) # Config 파일을 한줄씩 읽어와서 Inform
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID # CUDA_VISIBLE device Setup

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, _, _, _, _ = make_dataloader(cfg)
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
