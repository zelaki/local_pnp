import torch
import numpy as np
import tensorly as tl
from ldm.models.diffusion.ddim import DDIMSampler

from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image



def mapRange(value, inMin, inMax, outMin, outMax):
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))

def plot_masks(Us, r, s, rs=256, save_path=None, title_factors=True):
    """
    Plots the parts factors with matplotlib for visualization

    Parameters
    ----------
    Us : np.array
        Learnt parts factor matrix.
    r : int
        Number of factors to show.
    s : int
        Dimensions of each part (h*w).
    rs : int
        Target size to downsize images to.
    save_path : bool
        Save figure?
    title_factors : bool
        Print matplotlib title on each part?
    """


    fig = plt.figure(constrained_layout=True, figsize=(20, 3))
    spec = gridspec.GridSpec(ncols=r + 1, nrows=1, figure=fig)

    for i in range(0, r):
        fig.add_subplot(spec[i])

        if title_factors:
            plt.title(f'Part {i}')

        part = Us[i].reshape([s, s])
        part = mapRange(part, torch.min(part), torch.max(part), 0.0, 1.0) * 255
        part = part.detach().cpu().numpy()
        part = np.array(Image.fromarray(np.uint8(part)).convert('RGBA').resize((rs, rs), Image.NEAREST)) / 255

        plt.axis('off')
        plt.imshow(part, vmin=1, vmax=1, cmap='gray', alpha=1.00)

    if save_path is not None:
        plt.savefig(save_path)

import os
from tqdm import tqdm
from omegaconf import OmegaConf
import json
from run_features_extraction import load_model_from_config

def load_features_multi_exp(layer):
    model_config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(model_config, f"models/ldm/stable-diffusion-v1/model.ckpt")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)
    feature_maps = []
    for i in range(85):
        for t in iterator:
            feature_maps_path = f"/home/theokouz/src/plug-and-play/experiments/young_man_{i}/feature_maps"
            feature_maps.append(torch.load(os.path.join(feature_maps_path, f"output_block_{layer}_out_layers_features_time_{t}.pt"))[1].unsqueeze(0))
   
    return feature_maps


def load_feats(feature_maps_path, layer_num):

    model_config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(model_config, f"models/ldm/stable-diffusion-v1/model.ckpt")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

    feature_maps = []
    for t in iterator:
        feature_maps.append(torch.load(os.path.join(feature_maps_path, f"output_block_{layer_num}_out_layers_features_time_{t}.pt"))[1].unsqueeze(0))
   
    return feature_maps

def HOSVD(feature_maps, c, s, batch_size=10, n_iters=10):
    """
    Initialises the appearance basis A. In particular, computes the left-singular vectors of the channel mode's scatter matrix.

    Note: total samples used is batch_size * n_iters

    Parameters
    ----------
    batch_size : int
        Number of activations to sample in a single go.
    n_iters : int
        Number of times to sample `batch_size`-many activations.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    with torch.no_grad():
        Z = torch.zeros((batch_size * n_iters, c, s, s), device=device)

        # note: perform in loops to have a larger effective batch size
        print('Starting loops...')
        for i in range(n_iters):
            np.random.seed(i)
            torch.manual_seed(i)

            z = feature_maps[i]
            Z[(batch_size * i):(batch_size * (i + 1))] = z

        Z = Z.view([-1, c, s**2])
        print(f'Generated {batch_size * n_iters} gan samples...')
        # print(Z.shape)
        scat = 0
        for _, x in enumerate(Z):
            print(x.shape)
            # mode-3 unfolding in the paper, but in PyTorch channel mode is first.
            m_unfold = tl.unfold(x, 0)
            scat += m_unfold @ m_unfold.T

        Uc_init, _, _ = np.linalg.svd((scat / len(Z)).cpu().numpy())
        Uc_init = torch.Tensor(Uc_init).to(device)

        return Uc_init






def decompose_autograd(
        features,
        c,
        s,
        Uc_init=None, 
        ranks=[50, 8],
        lr=1e-5,
        batch_size=1,
        its=20000,
        log_modulo=1000,
        verbose=True,
        output_dir="./masks"
    ):
    """
    Performs the same decomposition in the paper, only uses autograd with Adam optimizer (and projected gradient descent).

    Parameters
    ----------
    ranks : list
        List of integers specifying the R_C and R_S, the ranks--i.e. number of parts and appearances respectively.
    lr : float
        Learning rate the projected gradient descent.
    batch_size : int
        Number of samples in each batch.
    its : int
        Total number of iterations.
    log_modulo : int
        Parameter used to control how often "training" information is displayed.
    hosvd_init : bool
        Initialise appearance factors from HOSVD? (else from random normal).
    verbose : bool
        Prints extra information.
    """
    ranks = ranks
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    #######################
    # init from HOSVD, else random normal
    Uc = torch.nn.Parameter(Uc_init[:, :ranks[0]].detach().clone().to(device), requires_grad=True) \
        if Uc_init is not None else torch.nn.Parameter(torch.randn(1280, ranks[0]).detach().clone().to(device) * 0.01)



    Us = torch.nn.Parameter(torch.Tensor(np.random.uniform(0, 0.01, size=[s**2, ranks[1]])).to(device), requires_grad=True)
    #######################
    optimizerS = torch.optim.Adam([Us], lr=lr)
    optimizerC = torch.optim.Adam([Uc], lr=lr)

    print(f'Uc shape: {Uc.shape}, Us shape: {Us.shape}')

    zeros = torch.zeros_like(Us, device=device)
    for t in range(its):
        np.random.seed(t)
        torch.manual_seed(t)


        # Z = features[t%25]
        Z = features[t%85]
        # print(Z.shape)
        # Update S
        # reconstruct
        coords = tl.tenalg.multi_mode_dot(Z.view(-1, c, s**2).float(), [Uc.T, Us.T], transpose=False, modes=[1, 2])
        Z_rec = tl.tenalg.multi_mode_dot(coords, [Uc, Us], transpose=False, modes=[1, 2])

        rec_loss = torch.mean(torch.norm(Z.view(-1, c, s**2).float() - Z_rec, p='fro', dim=[1, 2]) ** 2)
        rec_loss.backward(retain_graph=True)

        optimizerS.step()
        # --- projection step ---
        Us.data = torch.maximum(Us.data, zeros)
        optimizerS.zero_grad()
        optimizerC.zero_grad()

        # Update C
        # reconstruct with updated Us
        coords = tl.tenalg.multi_mode_dot(Z.view(-1, c, s**2).float(), [Uc.T, Us.T], transpose=False, modes=[1, 2])
        Z_rec = tl.tenalg.multi_mode_dot(coords, [Uc, Us], transpose=False, modes=[1, 2])

        rec_loss = torch.mean(torch.norm(Z.view(-1, c, s**2).float() - Z_rec, p='fro', dim=[1, 2]) ** 2)
        rec_loss.backward()
        optimizerC.step()
        optimizerS.zero_grad()
        optimizerC.zero_grad()

        Us = Us
        Uc = Uc

        with torch.no_grad():
            if t % log_modulo == 0 and verbose:
                print(f'Iteration {t} -- rec {rec_loss}')

                plot_masks(Us.T, r=min(ranks[-1], 32), s=s)
                plt.savefig(f"{output_dir}/mask_{t}")
    with torch.no_grad():

        torch.save(Uc, f"{output_dir}/Uc.pt")
        torch.save(Us, f"{output_dir}/Us.pt")



def refine(features, Us, Uc, s=16, ranks=[10, 16], lr=1e-8, its=5000, log_modulo=250, verbose=True):


    np.random.seed(0)
    torch.manual_seed(0)

    #######################
    # init from global spatial factors
    UsR = Us.clone().float()

    print(Us.shape)
    print(Uc.shape)

    #######################

    zeros = torch.zeros_like(Us, device="cuda")
    for t in range(its):
        with torch.no_grad():
            z = features[t%50].view(1, 1280, -1).float()
            # descend refinement term's gradient
            UsR_g = -4 * (torch.transpose(z,1,2)@Uc@Uc.T@z@UsR) + \
                2 * (UsR@UsR.T@torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@UsR + torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@UsR@UsR.T@UsR)
            UsR_g = torch.sum(UsR_g, 0)

            # Update S
            UsR = UsR - lr * UsR_g
            # PGD step
            UsR = torch.maximum(UsR, zeros)

            if ((t + 1) % log_modulo == 0 and verbose):
                print(f'iteration {t}')

                plot_masks(UsR.T, s=s, r=min(ranks[-1], 16))
                plt.show()
                plt.savefig(f"mask_{t}")
                # plot_colours(image, UsR.T, s=s, r=ranks[-1], seed=-1, alpha=0.9)
                # plt.show()

    return UsR





if __name__ == "__main__":
    import sys
    tl.set_backend('pytorch')

    # exp_name = sys.argv[1]
    # layer_num = sys.argv[2]
    layer_num=7

    # features_maps_path = f'experiments/young_man_0/feature_maps/'


    # example_feature_map = f"{features_maps_path}/output_block_{layer_num}_out_layers_features_time_1.pt"
    # if not os.path.exists(example_feature_map):
    #     print(example_feature_map)
    #     print(f"Feature maps for layer {layer_num} in experiment do not exist!")
    #     exit(1)
    # example_feature_map = torch.load(example_feature_map)
    # features_map_shape = example_feature_map.shape
    # channels = features_map_shape[1]
    # width = features_map_shape[2]
    # height = features_map_shape[3]



    # channels = 1280
    # width = 16
    # height = 16
    channels = 640
    width = 32
    height = 32



    # exp_name="young_man_multiseed_part_16"
    # output_dir = f'./masks/{exp_name}_{layer_num}'
    # features = load_features_multi_exp(layer=layer_num)
    # # output_dir = f'./masks/{exp_name}_{layer_num}'
    # # print(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    # # features = load_feats(feature_maps_path=features_maps_path, layer_num=layer_num)
    # init = HOSVD(feature_maps=features, c=channels, s=height)
    # decompose_autograd(features=features, c=channels, s=height, Uc_init=init, output_dir=output_dir)
    feats = load_feats(sys.argv[1], layer_num=4)
    Uc = torch.load("/home/theokouz/src/plug-and-play/masks/young_man_multiseed_part_16_4/Uc.pt")
    Us = torch.load("/home/theokouz/src/plug-and-play/masks/young_man_multiseed_part_16_4/Us.pt")
    Us = refine(features=feats, Uc=Uc, Us=Us)
    torch.save(Us, "/home/theokouz/src/plug-and-play/masks/young_man_multiseed_part_16_4/young_man_42/Usr_42.pt")