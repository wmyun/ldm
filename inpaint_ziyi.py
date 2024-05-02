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
from ldm.data.OAI import PainInpaintDataset
from ldm.data.leather import LeatherGlue
from ldm.data.defectall import Selected

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

    # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    # images = [x.replace("_mask.png", ".png") for x in masks]
    # print(f"Found {len(masks)} inputs.")
    # pain_dataset = PainInpaintDataset(data_root='/media/ziyi/Dataset/OAI_pain/full/ap/*', mode='eval')
    # eval_dataloader = data.DataLoader(dataset=pain_dataset, batch_size=1,
    #                                    num_workers=10, drop_last=True)
    # pain_dataset = LeatherGlue(data_root='/media/ziyi/Dataset/OAI_pain/full/ap/*', mode='test')
    pain_dataset = Selected(data_root='/media/ziyi/Dataset/OAI_pain/full/ap/*', mode='test')
    eval_dataloader = data.DataLoader(dataset=pain_dataset, batch_size=1,
                                       num_workers=10, drop_last=True)

    config = OmegaConf.load("models/ldm/defectall/config.yaml")

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/defectall/last.ckpt")["state_dict"],
                          strict=True)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model = model.to(device).eval()
    sampler = DDIMSampler(model)


    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for ret in tqdm(eval_dataloader):
                image = ret['image'].permute(0, 3, 1, 2).to(device=device)
                mask = ret['mask'].permute(0, 3, 1, 2).to(device=device)
                mask_img = ret['masked_image'].permute(0, 3, 1, 2).to(device=device)
                to_prog_mask = ret["to_prog_mask"].unsqueeze(1)to(device=device)


                defect_type = ret["defect_type"]
                outpath = os.path.join(opt.outdir, ret['relative_file_path_'][0])
                batch = {"image": image, "mask": mask, "masked_image": mask_img}

                c = model.get_learned_conditioning(batch["masked_image"])
                recon = model.cond_stage_model.decode(c)
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:], mode='nearest')
                to_prog_mask_inter = torch.nn.functional.interpolate(to_prog_mask,
                                                     size=c.shape[-2:], mode='nearest')
                # cc = (cc > 0).float()
                c = torch.cat((c, cc), dim=1)

                encoder_posterior = model.encode_first_stage(batch["image"])
                img_feature = model.get_first_stage_encoding(encoder_posterior).detach()

                shape = (int(c.shape[1]/2),)+c.shape[2:]

                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False,
                                                 x0=img_feature,
                                                 mask=cc,
                                                 ori_mask=to_prog_mask_inter)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                # prog_img, progressives = model.progressive_denoising(c,
                #                                            shape=shape,
                #                                            batch_size=1, mask=to_prog_mask_inter, x0=img_feature)
                # prog_row = model._get_denoise_row_from_list(progressives, desc="Progressive Generation")
                # prog_row = torch.clamp((prog_row+1.0)/2.0,
                #                        min=0.0, max=1.0)
                # prog_row = (prog_row.cpu().numpy() * 255).astype(np.uint8)
                # tiff.imwrite(outpath.replace('.png', f'_{defect_type}_prog.png'), prog_row)


                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp(to_prog_mask,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)
                recon = torch.clamp((recon+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = (inpainted.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                tiff.imwrite(outpath, inpainted)

                predicted_image = (predicted_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                tiff.imwrite(outpath.replace('.png', f'_{defect_type}pred.png'), predicted_image)

                recon = (recon.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                tiff.imwrite(outpath.replace('.png', f'_{defect_type}recon.png'), recon)

                recon = model.decode_first_stage(img_feature)
                recon = torch.clamp((recon+1.0)/2.0,
                                              min=0.0, max=1.0)
                recon = (recon.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                tiff.imwrite(outpath.replace('.png', f'_{defect_type}_recon_ori.png'), recon)

                img = batch["image"]
                img = (img + 1) / 2
                img = (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                tiff.imwrite(outpath.replace('.png', f'_{defect_type}_img.png'), img)

                cc = (cc.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                img = torch.nn.functional.interpolate(torch.Tensor(img).unsqueeze(0),
                                                     size=c.shape[-2:], mode='nearest').squeeze(0)
                # masked_img = np.array(img) * cc
                # cc = np.concatenate([cc, masked_img, np.array(img)], 2)
                # tiff.imwrite(outpath.replace('.png', f'_{defect_type}_cc.png'), cc)

# CUDA_VISIBLE_DEVICES=3 python inpaint.py --indir x --outdir /home/ziyi/Projects/latent-diffusion/1225_out