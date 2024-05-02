from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import os
import tifffile as tiff
import numpy as np

gt_path = "/home/meng-yun/Projects/rb_vessel/Dataset/vessel1_yz"
pred_path = "/home/meng-yun/Projects/latent/results/240311_50s"
data_len = 50
gt_list = sorted(os.listdir(gt_path))[0:data_len]
pred_list = sorted(os.listdir(pred_path))[0:data_len]

psnr = []
ssim = []
for i in range(data_len):
    gt_img = tiff.imread('{}/{}'.format(gt_path, gt_list[i]))
    pred_img = cv2.imread('{}/{}'.format(pred_path, pred_list[i]))[:,:,0]
    # print(pred_img.shape)
    # assert 0
    gt_img = (gt_img-gt_img.min())/(gt_img.max()-gt_img.min())*255
    gt_img = gt_img.astype(np.uint8)
    # pred_img = (pred_img-pred_img.min())/(pred_img.max()-pred_img.min())*255
    # pred_img = pred_img.astype(np.uint8)

    psnr.append(peak_signal_noise_ratio(gt_img,pred_img))
    ssim.append(structural_similarity(gt_img,pred_img))

psnr = np.asarray(psnr)
ssim = np.asarray(ssim)
print("PSNR(min, max, avg):",psnr.min(),psnr.max(),psnr.mean())
print("SSIM(min, max, avg):",ssim.min(),ssim.max(),ssim.mean())