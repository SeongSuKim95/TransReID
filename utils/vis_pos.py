import torch
import math

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

model_path = './pretrain/jx_vit_base_p16_224-80ecf9dd.pth' 

param_dict = torch.load(model_path, map_location='cpu')
for k, v in param_dict.items():
     if k == 'pos_embed' :
     # model_path = ./pretrain/jx_vit_base_p16_224-80ecf9dd.pth , pos_embed.shape = [1,197,768]
     # v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x) # linear interpolate positional embeddings
        print(v)
        print(v.shape)