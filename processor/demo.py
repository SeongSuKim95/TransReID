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

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument("--config_file", default="", help="path to config file", type=str)
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
# parser.add_argument('--test_dir',default='/mnt/hdd_data/Dataset/market1501',type=str, help='./test_data')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != "":
        cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

# data_dir = opts.test_dir
# image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}
_, _, _, _, _, _, _, query_loader, gallery_loader = make_dataloader(cfg)

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
#query_root = result['query_root'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
#gallery_root = result['query_root'][0]

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

i = args.query_index
index,score = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
# query_feature[i].size() = 768
# gallery_feature.size() = [15913,768]

########################################################################
# Visualize the rank result

query_path = query_loader.dataset.dataset[i][0]
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4)) #단위 인치
    ax = plt.subplot(1,11,1) # row, col, index
    ax.axis('off')
    imshow(query_path,'Query')
    ax.text(10,140,f"ID : {query_label}")
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        img_path = gallery_loader.dataset.dataset[index[i]][0] 
        label = gallery_label[index[i]]
        similarity_score = score[i]
        imshow(img_path)
        
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        ax.set
        ax.text(10,140,f"ID : {label}",) 
        ax.text(0,152,"Score : {:.3f}".format(similarity_score))
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = gallery_loader.dataset.dataset[index[i]][0]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
