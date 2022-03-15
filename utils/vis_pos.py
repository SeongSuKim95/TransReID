import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

def axis_off(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing positional embedding similarity")
    parser.add_argument(
        "--weight_dir", default="", help="path to weight file", type=str
    )
    parser.add_argument(
        "--patch_row", default="", help="Row number of patches", type=int
    )
    parser.add_argument(
        "--patch_col", default="", help="col number of patches", type=int
    )
    args = parser.parse_args()

    model_path = args.weight_dir
    weight_name = (model_path.split('/')[-1]).split('.')[0]
    param_dict = torch.load(model_path, map_location='cpu')
    for k, v in param_dict.items():
        if k == 'pos_embed' :
        # model_path = ./pretrain/jx_vit_base_p16_224-80ecf9dd.pth , pos_embed.shape = [1,197,768]
        # v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x) # linear interpolate positional embeddings
            pos_embed = v

    similarity_score = cosine_similarity(pos_embed[0],pos_embed[0])
    similarity = similarity_score[1:,1:]
    patch_num = similarity.shape[0]
    row = args.patch_row
    col = args.patch_col
    fig = plt.figure(figsize = (row,col))
    fig.suptitle(weight_name) 
    for i in range(patch_num):
        ax = plt.subplot(row,col,i+1)
        axis_off(ax)
        plt.imshow(similarity[i].reshape(row,col))
        ax.text(2.5,-1.5,f'({int(i/row)+1},{int(i%row)+1})',fontsize=12)
    plt.subplots_adjust(hspace=0.3)

    fig.savefig(f"pos_vis/{weight_name}.png")
