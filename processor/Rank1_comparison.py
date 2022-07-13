import argparse
import scipy.io
import torch
import numpy as np
import os
from config import cfg as cfg_1
from config import cfg_test as cfg_2 

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
from datasets.cuhk03np import Cuhk03np

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

def imwrite(path, title=None):
    """Imwrite for Tensor."""
    
    im = plt.imread(path)
    cv2.imwrite(f"{idx}.jpg",im)
    
def axis_off(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'cuhk03' : Cuhk03np,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
} # Class Factory

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument("--config_file1", default="", help="path to config file1", type=str)
parser.add_argument("--config_file2", default="", help="path to config file2", type=str)
parser.add_argument('--use_cuda', action='store_true', default=False,help='Use NVIDIA GPU acceleration')

# parser.add_argument('--test_dir',default='/mnt/hdd_data/Dataset/market1501',type=str, help='./test_data')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file1 != "":
        cfg_1.merge_from_file(args.config_file1)
cfg_1.merge_from_list(args.opts)
cfg_1.freeze()


if args.config_file2 != "":
        cfg_2.merge_from_file(args.config_file2)
cfg_2.merge_from_list(args.opts)
cfg_2.freeze()


transform = T.Compose([
    T.Resize(cfg_1.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg_1.INPUT.PIXEL_MEAN, std=cfg_1.INPUT.PIXEL_STD)
])

os.environ['CUDA_VISIBLE_DEVICES'] = cfg_1.MODEL.DEVICE_ID
WEIGHT1 = '/home/sungsu21/TransReID/TransReID/logs/cuhk03_vit_base_384_128/802/transformer_120.pth'
WEIGHT2 = '/home/sungsu21/TransReID/TransReID/logs/cuhk03_vit_base_384_128/801/transformer_120.pth'
INDEX1 = '802'
INDEX2 = '801'
idx = 4
data = cfg_1.DATASETS.NAMES
dataset = __factory[cfg_1.DATASETS.NAMES](root=cfg_1.DATASETS.ROOT_DIR)


model1 = make_model(cfg_1, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
model1.load_param(WEIGHT1)
model1.eval()

model2= make_model(cfg_2, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
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
rank = 1
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
        # dist_score_idx1 = dist_score1[idx][dist_mask1]

        dist_index_idx2 = dist_index2[idx][dist_mask2]
        # dist_score_idx2 = dist_score2[idx][dist_mask2]

        dist_rank_index_1[idx] = dist_index_idx1[0]
        dist_rank_index_2[idx] = dist_index_idx2[0]
        if (gl[dist_rank_index_1[idx]] == ql[idx]) and (gl[dist_rank_index_2[idx]] != ql[idx]) : # Model 1에선 맞췄으나 Model 2 에서 틀린 경우
            rank1_list_1[idx] = True
        if (gl[dist_rank_index_2[idx]] == ql[idx]) and (gl[dist_rank_index_1[idx]] != ql[idx]) : # Model 2에선 맞췄으나 Model 1 에서 틀린 경우
            rank1_list_2[idx] = True
        
        # dist_rank_score_1[idx] = dist_score_idx1[0]
        # dist_rank_score_2[idx] = dist_score_idx2[0]
    
    r1_list_1to1 = dist_rank_index_1[rank1_list_1]
    r1_list_2to2 = dist_rank_index_2[rank1_list_2]
    r1_list_1to2 = dist_rank_index_2[rank1_list_1]
    r1_list_2to1 = dist_rank_index_1[rank1_list_2]    

    return rank1_list_1, rank1_list_2, r1_list_1to1, r1_list_2to2, r1_list_1to2, r1_list_2to1


def Attention_map(img_path,model,cam_label=None):
    img = Image.open(img_path)
    img_input = img.resize(cfg_1.INPUT.SIZE_TEST)
    input_tensor = transform(img_input).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    
    # Attention map for model 1
    attention_rollout = VITAttentionRollout(model, head_fusion=cfg_1.TEST.HEAD_FUSION, 
    discard_ratio=cfg_1.TEST.DISCARD_RATIO)
    mask = attention_rollout(input_tensor,cam_label)
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) # If we use plt.show() in further process

    return mask

###########################
#Visualize the rank result#
###########################

R1_q_1,R1_q_2,R1_1to1,R1_2to2,R1_1to2,R1_2to1 = rank_1_idx(dist_eucd_1,dist_eucd_2,query_label,query_cam,gallery_label,gallery_cam)

# For model 1 --> 2 (Query, Rank1 gallery in 1, Rank1 gallery in 2)

R1_q_1_path = query_path_list[R1_q_1]
R1_q_1_label = query_label[R1_q_1]
R1_q_1_cam = query_cam[R1_q_1]

gallery_path_1to1 = gallery_path_list[R1_1to1]
gallery_path_1to2 = gallery_path_list[R1_1to2]

gallery_label_1to1 = gallery_label[R1_1to1]
gallery_label_1to2 = gallery_label[R1_1to2]

gallery_cam_1to1 = gallery_label[R1_1to1]
gallery_cam_1to2 = gallery_label[R1_1to2]

# For model 2 --> 1 (Query, Rank1 gallery in 2, Rank1 gallery in 1)

# R1_q_2_path = query_path_list[R1_q_2]
# R1_q_2_label = query_label[R1_q_2]
# R1_q_2_cam = query_cam[R1_q_2]

# gallery_path_2to1 = gallery_path_list[R1_2to1]
# gallery_path_2to2 = gallery_path_list[R1_2to2]

# gallery_label_2to1 = gallery_label[R1_2to1]
# gallery_label_2to2 = gallery_label[R1_2to2]

# gallery_cam_2to1 = gallery_cam[R1_2to1]
# gallery_cam_2to2 = gallery_cam[R1_2to2]

num1 = len(R1_q_1_path)

# num2 = len(R1_q_2_path)
print(f"Print {num1} Images (POS: True, Tri: False) ")

# For INDEX 1
fig1 = plt.figure(figsize=(5,num1)) #단위 인치
fig1.suptitle(f'TEST_SIZE : {cfg_1.INPUT.SIZE_TEST}, METRIC: {cfg_1.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg_1.TEST.HEAD_FUSION}', fontsize=5) 
for i in range(num1):
    if i == idx:
        if data == 'msmt17':
            query_path = (q_root.replace('list_query.txt','mask_test_v2/') + R1_q_1_path[i].split('_')[0] + '/' + R1_q_1_path[i]).rstrip()
            query_label_i = R1_q_1_label[i]
            query_cam_i = R1_q_1_cam[i]
            gallery_path_1to1_i = (g_root.replace('list_gallery.txt','mask_test_v2/') +gallery_path_1to1[i].split('_')[0] + '/' + gallery_path_1to1[i]).rstrip()
            gallery_label_1to1_i = gallery_label_1to1[i]
            gallery_path_1to2_i = (g_root.replace('list_gallery.txt','mask_test_v2/') +gallery_path_1to2[i].split('_')[0] + '/' + gallery_path_1to2[i]).rstrip()
            gallery_label_1to2_i = gallery_label_1to2[i]
            print(f"{i}th image : {query_path}")
        else :
            query_path = (q_root +'/'+ R1_q_1_path[i]).rstrip()
            query_label_i = R1_q_1_label[i]
            query_cam_i = R1_q_1_cam[i]

            gallery_path_1to1_i = (g_root + '/' + gallery_path_1to1[i]).rstrip()
            gallery_label_1to1_i = gallery_label_1to1[i]
            gallery_path_1to2_i = (g_root + '/' + gallery_path_1to2[i]).rstrip()
            gallery_label_1to2_i = gallery_label_1to2[i]
            print(f"{i}th image : {query_path}")

        # Query Image (Model 1 : True, Model2 : False)
        query_ax = plt.subplot(num1,7,7*i+1) # row, col, index
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        axis_off(query_ax)
        imshow(query_path)
        if i == 0 :
            plt.title('Query',fontsize=6)
        
        axis_set(query_ax, color="black",linewidth=1)
        # ax.text(10,140,f"ID : {query_label_i}")
        # ax.text(10,152,f"Cam : {query_cam_i}")
        query_1to1 = Attention_map(query_path,model1)
        query_1to2 = Attention_map(query_path,model2)
        gallery_1to1 = Attention_map(gallery_path_1to1_i,model1)
        gallery_1to2 = Attention_map(gallery_path_1to2_i,model2)
        
        # Attention map of Query for MODEL 1 
        attn_ax1 = plt.subplot(num1,7,7*i+2)
        axis_off(attn_ax1)
        plt.imshow(query_1to1)
        axis_set(attn_ax1, color="black",linewidth=1)

        # Rank1 Gallery Image for MODEL 1
        gallery_ax = plt.subplot(num1,7,7*i+3)
        axis_off(gallery_ax)
        imshow(gallery_path_1to1_i)
        axis_set(gallery_ax, color = "green", linewidth=1)
        # gallery_ax.set_xlabel(f'ID:{gallery_label_1to1_i}')
        plt.title(f'ID : {gallery_label_1to1_i}',fontsize = 6)

        # Attention map of Gallery Image for MODEL 1
        attn_ax2 = plt.subplot(num1,7,7*i+4)
        axis_off(attn_ax2)
        plt.imshow(gallery_1to1)
        axis_set(attn_ax2, color = "green", linewidth=1)
        # Attention map of Query for MODEL 2

        attn_ax3 = plt.subplot(num1,7,7*i+5)
        axis_off(attn_ax3)
        plt.imshow(query_1to2)
        axis_set(attn_ax3, color = "black", linewidth=1)

        # Rank 1 Gallery Image for MODEL 2
        gallery_ax = plt.subplot(num1,7,7*i+6)
        axis_off(gallery_ax)
        axis_set(gallery_ax, color = "red", linewidth=1)
        imshow(gallery_path_1to2_i)
        # gallery_ax.set_xlabel(f'ID:{gallery_label_1to1_i}')
        plt.title(f'ID : {gallery_label_1to2_i}',fontsize = 6)

        # Attention map of Gallery Image for MODEL 2
        attn_ax4 = plt.subplot(num1,7,7*i+7)
        axis_off(attn_ax4)
        axis_set(attn_ax4, color = "red", linewidth=1)
        plt.imshow(gallery_1to2)

        # Sample Extraction
        im = plt.imread(query_path)
        print(query_path)
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_query.jpg",cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        im = plt.imread(gallery_path_1to1_i)
        print(gallery_path_1to1_i)
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_gallery_path_1to1.jpg",cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        
        im = plt.imread(gallery_path_1to2_i)
        print(gallery_path_1to2_i)
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_gallery_path_1to2.jpg",cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_query_1to1.jpg",cv2.cvtColor(query_1to1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_gallery_1to1.jpg",cv2.cvtColor(gallery_1to1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}/{idx}_gallery_1to2.jpg",cv2.cvtColor(gallery_1to2, cv2.COLOR_RGB2BGR))

    # plt.title(f'{INDEX2}')
    
plt.subplots_adjust(wspace =0.02,hspace=0.3)
result_img_path = f'result/result_visualize/R1_comparison/{INDEX1}&{INDEX2}'
os.makedirs(result_img_path,exist_ok=True)
fig1.savefig(f"{result_img_path}/{INDEX1}&{INDEX2}_R1comparison_{cfg_1.TEST.HEAD_FUSION}_{cfg_1.TEST.DISCARD_RATIO}.png")

# For INDEX 2

# fig2 = plt.figure(figsize=(1.5,num2)) #단위 인치
# fig2.suptitle(f'TEST_SIZE : {cfg.INPUT.SIZE_TEST}, METRIC: {cfg.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg.TEST.HEAD_FUSION}', fontsize=8) 
# for i in range(num2):
#     query_path = (q_root +'/'+ R1_q_2_path[i]).rstrip()
#     query_label_i = R1_q_2_label[i]
#     query_cam_i = R1_q_2_cam[i]
#     print(f"{i}th image : {query_path}")
#     ax = plt.subplot(num2,3,3*i+1) # row, col, index
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
#     axis_off(ax)
#     imshow(query_path,'Query')
#     # ax.text(10,140,f"ID : {query_label_i}")
#     # ax.text(10,152,f"Cam : {query_cam_i}")

#     mask1,mask2 = Attention_map(query_path,model2,model1)
    
#     attn_ax1 = plt.subplot(num2,3,3*i+2)
#     axis_off(attn_ax1)
#     plt.imshow(mask1)
#     plt.title(f'{INDEX2}')
#     attn_ax2 = plt.subplot(num2,3,3*i+3)
#     axis_off(attn_ax2)
#     plt.imshow(mask2)
#     plt.title(f'{INDEX1}')
    
# plt.subplots_adjust(wspace = 0.05,hspace=0.05)
# result_img_path = f'result/result_visualize/R1_comparision/{INDEX1}&{INDEX2}'
# os.makedirs(result_img_path,exist_ok=True)
# fig2.savefig(f"{result_img_path}/{INDEX2}&{INDEX1}_R1comparision_{cfg.TEST.HEAD_FUSION}_{cfg.TEST.DISCARD_RATIO}.png")

