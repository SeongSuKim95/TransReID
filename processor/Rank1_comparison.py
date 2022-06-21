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
from .vit_rollout import VITAttentionRollout
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

WEIGHT1 = ''
WEIGHT2 = ''
INDEX1 = ''
INDEX2 = ''
dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

model1 = make_model(cfg, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
model1.load_param(WEIGHT1)
model1.eval()

model2= make_model(cfg, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
model2.load_param(WEIGHT2)
model2.eval()
 
######################################################################
result1 = scipy.io.loadmat(f'result/result_matrix/{INDEX1}.mat')
result2 = scipy.io.loadmat(f'result/result_matrix/{INDEX2}.mat')

query_feature_1 = torch.FloatTensor(result1['query_f']).cuda()
query_feature_2 = torch.FloatTensor(result2['query_f']).cuda()
gallery_feature_1 = torch.FloatTensor(result1['gallery_f']).cuda()
gallery_feature_2 = torch.FloatTensor(result2['gallery_f']).cuda()

dist_eucd_1 = result1['Euclidean_dist']
dist_cos_1 = result1['Cos_dist']

dist_eucd_2 = result2['Euclidean_dist']
dist_cos_2 = result2['Cos_dist']

query_cam = result1['query_cam'][0]
query_label = result1['query_label'][0]
q_root = result1['q_dir'][0]
query_path_list = result1['img_path'][:query_label.size]

gallery_cam = result1['gallery_cam'][0]
gallery_label = result1['gallery_label'][0]
g_root = result1['g_dir'][0]
gallery_path_list = result1['img_path'][query_label.size:]

#######################################################################
def sort_img_eucd(dist_eucd, index, ql, qc, gl, gc):
    
    eucd_score = dist_eucd[index]
    # predict index
    eucd_index = np.argsort(eucd_score) #sort by index from large to small
    eucd_score = np.sort(eucd_score)
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql[index]) # query label과 gallery index가 같은 index
    #same camera
    camera_index = np.argwhere(gc==qc[index]) # query cam과 gallery cam이 같은 index

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
    junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
    junk_index = np.append(junk_index2, junk_index1) 

    eucd_mask = np.in1d(eucd_index, junk_index, invert=True)
    
    eucd_index = eucd_index[eucd_mask]
    eucd_score = eucd_score[eucd_mask]

    return eucd_index, eucd_score

def sort_img_cos(dist_cos, index, ql, qc, gl, gc):
    
    cos_score = dist_cos[index]

    # predict index
    cos_index = np.argsort(cos_score)[::-1] #sort by index from large to small
    cos_score = np.sort(cos_score)[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql[index]) # query label과 gallery index가 같은 index
    #same camera
    camera_index = np.argwhere(gc==qc[index]) # query cam과 gallery cam이 같은 index

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
    junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
    junk_index = np.append(junk_index2, junk_index1) 

    cos_mask = np.in1d(cos_index, junk_index, invert=True) # 1차원 배열의 각 요소가 두번째 배열에도 있는지 확인

    cos_index = cos_index[cos_mask]
    cos_score = cos_score[cos_mask]
 
    return cos_index, cos_score

def rank_1_idx(dist_mat1, dist_mat2, ql, qc, gl, gc):
    
    query_length = dist_mat1.shape[0]
    query_ID = np.unique(ql)
    # predict index
    dist_index1 = np.argsort(dist_mat1)
    dist_score1 = np.sort(dist_mat1)

    dist_index2 = np.argsort(dist_mat2)
    dist_score2 = np.sort(dist_mat2)

    #cos_score = np.flip(cos_score,1) # sort by index from large to small
    dist_rank_index_1 = np.zeros((query_length)).astype(int)
    dist_rank_score_1 = np.zeros((query_length))
    rank1_list_1 = np.zeros((query_length)).astype(bool)

    dist_rank_index_2 = np.zeros((query_length)).astype(int)
    dist_rank_score_2 = np.zeros((query_length))
    rank1_list_2 = np.zeros((query_length)).astype(bool)

    for idx in range(query_length):
        query_index = np.argwhere(gl==ql[idx]) # query label과 gallery index가 같은 index
        #same camera
        camera_index = np.argwhere(gc==qc[idx]) # query cam과 gallery cam이 같은 index
        #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
        junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
        junk_index = np.append(junk_index2, junk_index1) 

        dist_mask1 = np.in1d(dist_index1[idx], junk_index, invert=True) # 1차원 배열의 각 요소가 두번째 배열에도 있는지 확인
        dist_mask2 = np.in1d(dist_index2[idx], junk_index, invert=True) 
        
        dist_index_idx1 = dist_index1[idx][dist_mask1]
        dist_score_idx1 = dist_score1[idx][dist_mask1]

        dist_index_idx2 = dist_index2[idx][dist_mask2]
        dist_score_idx2 = dist_score2[idx][dist_mask2]

        dist_rank_index_1[idx] = dist_index_idx1[0]
        dist_rank_index_2[idx] = dist_index_idx2[0]
        if (gl[dist_rank_index_1[idx]] == ql[idx]) and (gl[dist_rank_index_2[idx] != ql[idx]]) : # Model 1에선 맞췄으나 Model 2 에서 틀린 경우
            rank1_list_1[idx] = True
        if (gl[dist_rank_index_2[idx]] == ql[idx]) and (gl[dist_rank_index_1[idx] != ql[idx]]) : # Model 2에선 맞췄으나 Model 1 에서 틀린 경우
            rank1_list_2[idx] = True
        
        dist_rank_score_1[idx] = dist_score_idx1[0]
        dist_rank_score_2[idx] = dist_score_idx2[0]

    return rank1_list_1,rank1_list_2,dist_rank_score_1,dist_rank_score_2


def Attention_map(img_path,cam_label=None):
    img = Image.open(img_path)
    img_input = img.resize(cfg.INPUT.SIZE_TEST)
    input_tensor = transform(img_input).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    attention_rollout = VITAttentionRollout(model, head_fusion=cfg.TEST.HEAD_FUSION, 
    discard_ratio=cfg.TEST.DISCARD_RATIO)
    mask = attention_rollout(input_tensor,cam_label)
    # name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) # If we use plt.show() in further process

    return mask

# cv2.imwrite("./result/"+sname, mask)
# index,score = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
# query_feature[i].size() = 768
# gallery_feature.size() = [15913,768]

###########################
#Visualize the rank result#
###########################

R1_list_1,R1_list_2,dist_1,dist_2 = rank_1_idx(result1,result2,query_label,query_cam,gallery_label,gallery_cam)


if cfg.TEST.VISUALIZE_TYPE == 0 :
    query_path = (q_root +'/'+ query_path_list[i]).rstrip()
    query_label_i = query_label[i]
    query_cam_i = query_cam[i]
    print(query_path)
    print('Top 10 images are as follow:')
    fig = plt.figure(figsize=(16,8)) #단위 인치
    fig.suptitle(f'TEST_SIZE : {cfg.INPUT.SIZE_TEST}, METRIC: {cfg.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg.TEST.HEAD_FUSION}', fontsize=16) 
    ax = plt.subplot(2,rank+1,1) # row, col, index
    axis_off(ax)
    imshow(query_path,'Query')
    ax.text(10,140,f"ID : {query_label_i}")
    ax.text(10,152,f"Cam : {query_cam_i}")
    if cfg.MODEL.SIE_CAMERA:
        mask = Attention_map(query_path,query_cam_i)
    else:
        mask = Attention_map(query_path)
    attn_ax = plt.subplot(2,rank+1,rank+2)
    axis_off(attn_ax)
    plt.imshow(mask)
    plt.title('Query')
    
    if cfg.TEST.VISUALIZE_METRIC == 'Euclidean':
        index,score = sort_img_eucd(dist_eucd,i,query_label,query_cam,gallery_label,gallery_cam)
    elif cfg.TEST.VISUALIZE_METRIC == 'Cos' :
        index,score = sort_img_cos(dist_cos,i,query_label,query_cam,gallery_label,gallery_cam)
    else :
        raise NotImplementedError("Visualize metric should be Euclidean or Cosine similarity")
    for i in range(rank):
        ax = plt.subplot(2,rank+1,i+2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = (g_root + '/' + gallery_path_list[index[i]]).rstrip()
        label = gallery_label[index[i]]
        cam = gallery_cam[index[i]]
        similarity_score = score[i]
        imshow(img_path)
        if label == query_label_i:
            ax.set_title('%d'%(i+1), color='green')
            axis_set(ax,'green',4)
        else:
            ax.set_title('%d'%(i+1), color='red')
            axis_set(ax,'red',4)

        ax.text(10,140,f"ID : {label}",)
        ax.text(10,152,f"Cam : {cam}",)  
        ax.text(0,164,"Score : {:.3f}".format(similarity_score))
        
        ax = plt.subplot(2,rank+1,i+rank+2+1)
        axis_off(ax)
        if cfg.MODEL.SIE_CAMERA:
            mask = Attention_map(img_path,cam)
        else:
            mask = Attention_map(img_path)
        plt.imshow(mask)
        print(img_path)

    plt.subplots_adjust(hspace=0.01)
    result_img_path = f'result/result_visualize/{cfg.INDEX}'
    os.makedirs(result_img_path,exist_ok=True)
    fig.savefig(f"{result_img_path}/{cfg.TEST.VISUALIZE_TYPE}_{cfg.TEST.VISUALIZE_INDEX}_{cfg.TEST.HEAD_FUSION}_{cfg.TEST.DISCARD_RATIO}_{cfg.TEST.VISUALIZE_METRIC}.png")

