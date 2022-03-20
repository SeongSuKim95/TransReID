import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.manifold import TSNE

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    #dist_mat = np.arccos(dist_mat)
    return dist_mat

def visualize_similarity(cls_weight1, cls_weight2):
    similarity = cosine_similarity(cls_weight1,cls_weight2)
    fig = plt.figure(figsize = (16,16))
    plt.imshow(similarity)
    fig.savefig("dist_mat.png")

def visualize_TSNE(weight):

    model = TSNE(n_components = 2)
    z = model.fit_transform(weight)

    x = z[:,0]
    y = z[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(x.shape[0]):
        ax.scatter(x[i],y[i])
    fig.savefig("weight_TSNE.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weight params debugging")
    parser.add_argument(
        "--weight_dir1", default="", help="path to weight file", type=str
    )
    # parser.add_argument(
    #     "--weight_dir2", default="", help="path to weight file", type=str
    # )
    args = parser.parse_args()

    model_path1 = args.weight_dir1
    # model_path2 = args.weight_dir2
    
    param_dict1 = torch.load(model_path1, map_location='cpu')
    # param_dict2 = torch.load(model_path2, map_location='cpu')

    print(param_dict1['base.patch_embed.proj.weight'].shape)
    # weight_1 = param_dict1['classifier.weight']


    # weight_2 = param_dict2['classifier.weight']
    # for k, v in param_dict.items():
    #     if "classifier" in k:
    #         classifier_weight_1 = v

    # visualize_similarity(weight_1,weight_2)
    
