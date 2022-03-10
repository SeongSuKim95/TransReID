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

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
} # Class Factory

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

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
# data_dir = opts.test_dir
# image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}
#####################################################################
#Show result
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
######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
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

# img_path = result['img_path']

# multi = os.path.isfile('multi_query.mat')

# if multi:
#     m_result = scipy.io.loadmat('multi_query.mat')
#     mquery_feature = torch.FloatTensor(m_result['mquery_f'])
#     mquery_cam = m_result['mquery_cam'][0]
#     mquery_label = m_result['mquery_label'][0]
#     mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # query.view(-1,1) --> torch.size([768,1])
    # print(query.shape)
    score = torch.mm(gf,query) # gf.size([15913,768]), query.size([768,1]) --> matrix multiplication [15913,1]
    # 각 gallery feature와 query feature간 내적 (similairity)
    score = score.squeeze(1).cpu() # feature similarity size : gallery size
    score = score.numpy() 
    # predict index
    index = np.argsort(score)[::-1] #sort by index from large to small
    score = np.sort(score)[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql) # query label과 gallery index가 같은 index
    #same camera
    camera_index = np.argwhere(gc==qc) # query cam과 gallery cam이 같은 index

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1) # gallery id가 -1인 건 제외
    junk_index2 = np.intersect1d(query_index, camera_index) # np.intersect1d : 교집합, 즉 pid와 cid가 같은 gallery
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True) # 1차원 배열의 각 요소가 두번째 배열에도 있는지 확인
    index = index[mask]
    score = score[mask]
    # test = np.array([0, 1, 2, 5, 0])
    # states = [0, 2]
    # mask = np.in1d(test, states)
    # mask
    # array([ True, False,  True, False,  True])
    # test[mask]
    # array([0, 2, 0])
    # mask = np.in1d(test, states, invert=True)
    # mask
    # array([False,  True, False,  True, False])
    # test[mask]
    # array([1, 5])
    
    # 즉, junk index를 제외한 index를 return 하겠다는 것

    return index,score

def Attention_map(img_path):
    img = Image.open(img_path)
    img_input = img.resize(cfg.INPUT.SIZE_TEST)
    input_tensor = transform(img_input).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    attention_rollout = VITAttentionRollout(model, head_fusion=cfg.TEST.HEAD_FUSION, 
    discard_ratio=cfg.TEST.DISCARD_RATIO)
    mask = attention_rollout(input_tensor)
    # name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    # cv2.imshow("Input Image", np_img)
    # cv2.imshow(name, mask)
    # cv2.imwrite("input.png", np_img)
    
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB) # If we use plt.show() in further process

    return mask
    # cv2.imwrite("./result/"+sname, mask)
i = cfg.TEST.VISUALIZE_INDEX
index,score = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
# query_feature[i].size() = 768
# gallery_feature.size() = [15913,768]

########################################################################
# Visualize the rank result

query_path = (q_root +'/'+ query_path_list[i]).rstrip()
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,8)) #단위 인치
    ax = plt.subplot(2,11,1) # row, col, index
    axis_off(ax)
    imshow(query_path,'Query')
    ax.text(10,140,f"ID : {query_label}")
    mask = Attention_map(query_path)
    attn_ax = plt.subplot(2,11,12)
    axis_off(attn_ax)
    plt.imshow(mask)
    plt.title('Query')
    for i in range(10):
        ax = plt.subplot(2,11,i+2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = (g_root + '/' + gallery_path_list[index[i]]).rstrip()
        label = gallery_label[index[i]]
        similarity_score = score[i]
        imshow(img_path)
        
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')

        ax.text(10,140,f"ID : {label}",) 
        ax.text(0,152,"Score : {:.3f}".format(similarity_score))
        
        ax = plt.subplot(2,11,i+12+1)
        axis_off(ax)
        mask = Attention_map(img_path)
        plt.imshow(mask)
        print(img_path)
    plt.subplots_adjust(hspace=0.01)
    
except RuntimeError:
    for i in range(10):
        img_path = (g_root + '/' + gallery_path_list[index[i]]).rstrip()
        print(img_path)
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
