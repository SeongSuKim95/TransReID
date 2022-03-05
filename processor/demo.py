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
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = args.query_index
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result

query_path = query_loader.dataset.dataset[i][0]
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path = gallery_loader.dataset.dataset[index[i]][0]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = gallery_loader.dataset.dataset[index[i]][0]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
