import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import torch.utils.data as data
import tifffile as tiff

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.yztoxy import ZEnhanceDataset
import cv2
import time

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


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

    yz_dataset = ZEnhanceDataset(data_root=['/home/meng-yun/Projects/rb_vessel/Dataset/8x_yz_torchup/'], data_len=[0,50], 
                                    mask_config={"direction": "horizontal", "down_size": 8},
                                    image_size=512, mode='test')
    eval_dataloader = data.DataLoader(dataset=yz_dataset, batch_size=1,
                                       num_workers=10, drop_last=True)

    config = OmegaConf.load("logs/2024-04-27T11-59-39_yztoxy_ori/configs/2024-04-27T11-59-39-project.yaml")

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("logs/2024-04-27T11-59-39_yztoxy_ori/checkpoints/epoch=001873.ckpt")["state_dict"],
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
                image = ret['image'].to(device=device)
                cond_image = ret['cond_image'].to(device=device)
                batch = {"image": image, "cond_image": cond_image}
                
                # model.first_stage_model = model.first_stage_model.train()
                # model.cond_stage_model = model.cond_stage_model.train()
                
                c = model.get_learned_conditioning(batch["cond_image"])
                recon = model.cond_stage_model.decode(c)
                
                encoder_posterior = model.encode_first_stage(batch["image"])
                img_feature = model.get_first_stage_encoding(encoder_posterior).detach()
                #img_feature = encoder_posterior.detach()

                shape = img_feature.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
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

# CUDA_VISIBLE_DEVICES=3 python inpaint.py --indir x --outdir /home/ziyi/Projects/latent-diffusion/1225_out
# CUDA_VISIBLE_DEVICES=3 python yztoxy_test.py --outdir /home/meng-yun/Projects/latent/results/240322