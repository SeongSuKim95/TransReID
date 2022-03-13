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

dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
model = make_model(cfg, num_class=dataset.num_train_pids, camera_num=dataset.num_train_cams, view_num = dataset.num_train_vids)
model.load_param(cfg.TEST.WEIGHT)
model.eval()

######################################################################
result = scipy.io.loadmat(f'result/result_matrix/{cfg.INDEX}.mat')

query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
q_root = result['q_dir'][0]
query_path_list = result['img_path'][:query_label.size]

gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
g_root = result['g_dir'][0]
gallery_path_list = result['img_path'][query_label.size:]

dist_eucd = result['Euclidean_dist']
dist_cos = result['Cos_dist']

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
i = cfg.TEST.VISUALIZE_INDEX
rank = cfg.TEST.VISUALIZE_RANK

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

def max_rank_error(dist_mat, rank, ql, qc, gl, gc):
    
    query_length = dist_mat.shape[0]
    query_ID = np.unique(ql)
    # predict index
    dist_index = np.argsort(dist_mat)
    dist_score = np.sort(dist_mat)
    #cos_score = np.flip(cos_score,1) # sort by index from large to small
    dist_rank_index = np.zeros((query_length,rank)).astype(int)
    dist_rank_score = np.zeros((query_length,rank))

    query_ID_wise_match = np.zeros(query_ID.max()+1).astype(int)

    for idx in range(query_length):
        query_index = np.argwhere(gl==ql[idx]) # query label과 gallery index가 같은 index
        #same camera
        camera_index = np.argwhere(gc==qc[idx]) # query cam과 gallery cam이 같은 index
        #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
        junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
        junk_index = np.append(junk_index2, junk_index1) 

        dist_mask = np.in1d(dist_index[idx], junk_index, invert=True) # 1차원 배열의 각 요소가 두번째 배열에도 있는지 확인
        
        dist_index_idx = dist_index[idx][dist_mask]
        dist_score_idx = dist_score[idx][dist_mask]

        dist_rank_index[idx] = dist_index_idx[:rank]
        query_ID_wise_match[ql[idx]] += np.count_nonzero(gl[dist_rank_index[idx]] != ql[idx])
        dist_rank_score[idx] = dist_score_idx[:rank]
    
    query_ID_wise_match = query_ID_wise_match[query_ID]

    Max_error_ID = query_ID[np.argsort(query_ID_wise_match)[-1]]
    Max_error_ID_idx = np.where(ql==Max_error_ID)[0]
    Rank_error_idx = dist_rank_index[Max_error_ID_idx]
    Rank_error_score = dist_rank_score[Max_error_ID_idx]

    return Max_error_ID_idx, Rank_error_idx, Rank_error_score

def max_dist(dist, rank, ql, qc, gl, gc):
    
    query_length = dist.shape[0]
    # predict index
    dist_index = np.argsort(dist) # sort by index from large to small
    dist_score = np.sort(dist)
    # index = index[0:2000]
    # good index
    dist_rank_index = np.zeros((query_length,rank)).astype(int)
    dist_rank_score = np.zeros((query_length,rank))
    
    for idx in range(query_length):

        query_index = np.argwhere(gl==ql[idx]) # query label과 gallery index가 같은 index
        #same camera
        camera_index = np.argwhere(gc==qc[idx]) # query cam과 gallery cam이 같은 index
        #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
        junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
        junk_index = np.append(junk_index2, junk_index1) 

        dist_mask = np.in1d(dist_index[idx], junk_index, invert=True)
        
        dist_index_idx = dist_index[idx][dist_mask]
        dist_score_idx = dist_score[idx][dist_mask]

        dist_rank_index[idx] = dist_index_idx[:rank]
        dist_rank_score[idx] = dist_score_idx[:rank]

    dist_score_sum = np.sum(dist_rank_score,axis=1)
    dist_max_query = np.argsort(dist_score_sum)[::-1][:5]
    dist_max_index = dist_rank_index[dist_max_query] 
    dist_max_score = dist_rank_score[dist_max_query]
    
    return dist_max_query, dist_max_index, dist_max_score

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

elif cfg.TEST.VISUALIZE_TYPE == 1 :

    if cfg.TEST.VISUALIZE_METRIC == 'Euclidean':
        Rank_error_ID_idx, Rank_error_idx, Rank_error_score = max_rank_error(dist_eucd, rank, query_label, query_cam , gallery_label, gallery_cam)
    elif cfg.TEST.VISUALIZE_METRIC == 'Cos':
        Rank_error_ID_idx, Rank_error_idx, Rank_error_score = max_rank_error(dist_cos, rank, query_label, query_cam , gallery_label, gallery_cam)
    else :
        raise NotImplementedError("Visualize metric should be Euclidean or Cosine similarity")
    vis_num = len(Rank_error_ID_idx)
    fig = plt.figure(figsize=(16,4*vis_num*2))
    fig.suptitle(f'TEST_SIZE : {cfg.INPUT.SIZE_TEST}, METRIC: {cfg.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg.TEST.HEAD_FUSION}', fontsize=16) 
    for iter ,Query_idx in enumerate(Rank_error_ID_idx):
        ax = plt.subplot(vis_num*2,rank+1,2*iter*(rank+1)+1) # row, col, index
        axis_off(ax)
        query_path = (q_root +'/'+ query_path_list[Query_idx]).rstrip()
        query_label_iter = query_label[Query_idx]
        query_cam_iter = query_cam[Query_idx]
        imshow(query_path,'Query')
        ax.text(10,140,f"ID : {query_label_iter}")
        ax.text(10,152,f"Cam : {query_cam_iter}")
        if cfg.MODEL.SIE_CAMERA:
            mask = Attention_map(query_path,query_cam_iter)
        else:
            mask = Attention_map(query_path)        
        attn_ax = plt.subplot(vis_num*2,rank+1,(2*iter+1)*(rank+1)+1)
        axis_off(attn_ax)
        plt.imshow(mask)
        plt.title('Query')
        index = Rank_error_idx[iter]
        score = Rank_error_score[iter]
        for i in range(rank):
            ax = plt.subplot(vis_num*2,rank+1,i+2*iter*(rank+1)+2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_path = (g_root + '/' + gallery_path_list[index[i]]).rstrip()
            label = gallery_label[index[i]]
            cam = gallery_cam[index[i]]
            similarity_score = score[i]
            imshow(img_path)
            if label == query_label_iter:
                ax.set_title('%d'%(i+1), color='green')
                axis_set(ax,'green',4)
            else:
                ax.set_title('%d'%(i+1), color='red')
                axis_set(ax,'red',4)

            ax.text(10,140,f"ID : {label}",)
            ax.text(10,152,f"Cam : {cam}",)  
            ax.text(0,164,"Score : {:.3f}".format(similarity_score))
            
            ax = plt.subplot(vis_num*2,rank+1,i+(2*iter+1)*(rank+1)+2)
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
    fig.savefig(f"{result_img_path}/{cfg.TEST.VISUALIZE_TYPE}_{cfg.TEST.HEAD_FUSION}_{cfg.TEST.DISCARD_RATIO}_{cfg.TEST.VISUALIZE_METRIC}.png")

elif cfg.TEST.VISUALIZE_TYPE == 2 :
    
    if cfg.TEST.VISUALIZE_METRIC == 'Euclidean':
        max_dist_ID, max_dist_index, max_dist_score = max_dist(dist_eucd, rank, query_label, query_cam , gallery_label, gallery_cam)
    elif cfg.TEST.VISUALIZE_METRIC == 'Cos':
        max_dist_ID, max_dist_index, max_dist_score = max_dist(dist_cos, rank, query_label, query_cam , gallery_label, gallery_cam)
    else :
        raise NotImplementedError("Visualize metric should be Euclidean or Cosine similarity")
    vis_num = len(max_dist_ID)
    fig = plt.figure(figsize=(16,4*vis_num*2))
    fig.suptitle(f'TEST_SIZE : {cfg.INPUT.SIZE_TEST}, METRIC: {cfg.TEST.VISUALIZE_METRIC}, ATTENTION_VISUALIZE : {cfg.TEST.HEAD_FUSION}', fontsize=16) 
    for iter ,Query_idx in enumerate(max_dist_ID):
        ax = plt.subplot(vis_num*2,rank+1,2*iter*(rank+1)+1) # row, col, index
        axis_off(ax)
        query_path = (q_root +'/'+ query_path_list[Query_idx]).rstrip()
        query_label_iter = query_label[Query_idx]
        query_cam_iter = query_cam[Query_idx]
        imshow(query_path,'Query')
        ax.text(10,140,f"ID : {query_label_iter}")
        ax.text(10,152,f"Cam : {query_cam_iter}")
        if cfg.MODEL.SIE_CAMERA:
            mask = Attention_map(query_path,query_cam_iter)
        else:
            mask = Attention_map(query_path)             
        attn_ax = plt.subplot(vis_num*2,rank+1,(2*iter+1)*(rank+1)+1)
        axis_off(attn_ax)
        plt.imshow(mask)
        plt.title('Query')
        index = max_dist_index[iter]
        score = max_dist_score[iter]
        for i in range(rank):
            ax = plt.subplot(vis_num*2,rank+1,i+2*iter*(rank+1)+2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_path = (g_root + '/' + gallery_path_list[index[i]]).rstrip()
            label = gallery_label[index[i]]
            cam = gallery_cam[index[i]]
            similarity_score = score[i]
            imshow(img_path)
            if label == query_label_iter:
                ax.set_title('%d'%(i+1), color='green')
                axis_set(ax,'green',4)
            else:
                ax.set_title('%d'%(i+1), color='red')
                axis_set(ax,'red',4)
            ax.text(10,140,f"ID : {label}",)
            ax.text(10,152,f"Cam : {cam}",)  
            ax.text(0,164,"Score : {:.3f}".format(similarity_score))
            
            ax = plt.subplot(vis_num*2,rank+1,i+(2*iter+1)*(rank+1)+2)
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
        fig.savefig(f"{result_img_path}/{cfg.TEST.VISUALIZE_TYPE}_{cfg.TEST.HEAD_FUSION}_{cfg.TEST.DISCARD_RATIO}_{cfg.TEST.VISUALIZE_METRIC}.png")
