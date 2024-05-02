import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import torch.utils.data as data
import tifffile as tiff

from main import instantiate_from_config
from ldm.models.diffusion.pseudo_ddim import DDIMSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.data.yztoxy import ZEnhanceDataset
import cv2
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    yz_dataset = ZEnhanceDataset(data_root=['/home/meng-yun/Projects/rb_vessel/Dataset/8x_yz_torchup/','/home/meng-yun/Projects/rb_vessel/Dataset/8x_yz_torchup_pseudo'],
                                    data_len=[0,50], 
                                    mask_config={"direction": "horizontal", "down_size": 8},
                                    image_size=512, mode='test')
    eval_dataloader = data.DataLoader(dataset=yz_dataset, batch_size=1,
                                       num_workers=10, drop_last=True)

    config = OmegaConf.load("logs/2024-04-11T15-23-11_pcross_yztoxy/configs/2024-04-11T15-23-11-project.yaml")

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("logs/2024-04-11T15-23-11_pcross_yztoxy/checkpoints/epoch=001149.ckpt")["state_dict"],
                          strict=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu") # why using cpu??
    model = model.to(device).eval()
    sampler = DDIMSampler(model)


    os.makedirs(opt.outdir, exist_ok=True)
    s_t = time.time()
    with torch.no_grad():
        with model.ema_scope():
            for ret in tqdm(eval_dataloader):
                #image = ret['image'].to(device=device)
                cond_image = ret['cond_image'].to(device=device)
                pseudo = ret['pseudo'].to(device=device)
                batch = {"cond_image": cond_image, "pseudo": pseudo}#"image": image, 
                
                c = model.get_learned_conditioning(batch["pseudo"])
                #recon = model.cond_stage_model.decode(c)
                
                p = model.encode_first_stage(batch["cond_image"])
                if isinstance(p, DiagonalGaussianDistribution):
                    p = p.mode()

                shape = p.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 pseudo=p,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)#,
                                                 #x0=img_feature)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
 
                # image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                # recon = torch.clamp((recon+1.0)/2.0, min=0.0, max=1.0)
                
                outpath = os.path.join(opt.outdir, ret['file_path_'][0].replace('.tif', '.png'))
                predicted_image = (predicted_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(outpath.replace('.png', f'_pred.png'), np.transpose(np.concatenate([predicted_image, predicted_image, predicted_image], 0), (1, 2, 0)))

                # recon = (recon.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_recon.png'), np.transpose(np.concatenate([recon, recon, recon], 0), (1, 2, 0)))

                # recon = model.decode_first_stage(img_feature)
                # recon = torch.clamp((recon+1.0)/2.0,
                #                               min=0.0, max=1.0)
                # recon = (recon.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_recon_ori.png'), np.transpose(np.concatenate([recon, recon, recon], 0), (1, 2, 0)))

                # img = batch["image"]
                # img = (img + 1) / 2
                # img = (img.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.imwrite(outpath.replace('.png', f'_img.png'), np.transpose(np.concatenate([img, img, img], 0), (1, 2, 0)))
    e_t = time.time()
    print('Total time:', e_t-s_t)

# CUDA_VISIBLE_DEVICES=3 python p_yztoxy_test.py --outdir /home/meng-yun/Projects/latent/results/240408