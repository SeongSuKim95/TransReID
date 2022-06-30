import argparse
import scipy.io
import torch
import numpy as np
from config import cfg
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datasets import make_dataloader
from utils.logger import setup_logger
import cv2
from .vit_rollout import VITAttentionRollout_with_patches
from torchvision import transforms as T
from PIL import Image
from model import make_model

from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.veri import VeRi
from datasets.vehicleid import VehicleID

def axis_set(ax,color,linewidth):
    for pos in ['left','right','top','bottom']:
        ax.spines[pos].set_linewidth(linewidth)
        ax.spines[pos].set_color(color)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def axis_off(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
} # Class Factory

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument("--config_file", default="", help="path to config file", type=str)
parser.add_argument('--use_cuda', action='store_true', default=False,help='Use NVIDIA GPU acceleration')

# parser.add_argument('--test_dir',default='/mnt/hdd_data/Dataset/market1501',type=str, help='./test_data')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != "":
        cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
WEIGHT = '/home/sungsu21/TransReID/TransReID/logs/market_vit_base_384_128/408/transformer_120.pth'
INDEX = '408'
dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

model = make_model(cfg, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
model.load_param(WEIGHT)
model.eval()
 
def Attention_map(img_path,model,cam_label=None):
    img = Image.open(img_path)
    img_input = img.resize(cfg.INPUT.SIZE_TEST)
    input_tensor = transform(img_input).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    
    # Attention map for model 
    attention_rollout = VITAttentionRollout_with_patches(model, head_fusion=cfg.TEST.HEAD_FUSION, discard_ratio=cfg.TEST.DISCARD_RATIO)
    mask, patches = attention_rollout(input_tensor,cam_label)
    # np_img = np.array(img)[:, :, ::-1]
    # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    # mask = show_mask_on_image(np_img, mask)
    
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) # If we use plt.show() in further process

    return mask,patches

img_path = '/home/sungsu21/TransReID/TransReID/dataset/market1501/bounding_box_train/0002_c1s1_000451_03.jpg'
Attn_score, patches = Attention_map(img_path,model)
Attn_score = Attn_score.reshape(Attn_score.shape[0]*Attn_score.shape[1])

Patch_score = patches[0] @ patches[1:].T

Attn_sort = np.argsort(Attn_score)
Patch_sort = torch.sort(Patch_score)
print("d")
###########################
#Visualize the rank result#
###########################

# fig1 = plt.figure(figsize=(5,5)) #단위 인치
# fig1.suptitle(f'TEST_SIZE : {cfg.INPUT.SIZE_TEST}, METRIC: {cfg.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg.TEST.HEAD_FUSION}', fontsize=5)     
# plt.subplots_adjust(wspace =0.02,hspace=0.25)
# result_img_path = f'result/result_visualize/Class_patch/{img_path}'
# os.makedirs(result_img_path,exist_ok=True)
# fig1.savefig(f"{result_img_path}/ATTN_{cfg.TEST.HEAD_FUSION}_{cfg.TEST.DISCARD_RATIO}.png")
