# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np
import cv2
import os
from scipy import signal

from methods.Bicubic import Bicubic
from methods.Brovey import Brovey
from methods.PCA import PCA
from methods.IHS import IHS
from methods.SFIM import SFIM
from methods.GS import GS
from methods.Wavelet import Wavelet
from methods.MTF_GLP import MTF_GLP
from methods.MTF_GLP_HPM import MTF_GLP_HPM
from methods.GSA import GSA
from methods.CNMF import CNMF
from methods.GFPCA import GFPCA
from metrics import ref_evaluate, no_ref_evaluate
from PIL import Image

'''loading data'''
original_msi = np.array(Image.open('/ghome/fuxy/yaogan/WV2_data/test128/ms/788.tif'))
original_pan = np.array(Image.open('/ghome/fuxy/yaogan/WV2_data/test128/pan/788.tif'))

'''normalization'''
# max_patch, min_patch = np.max(original_msi, axis=(0,1)), np.min(original_msi, axis=(0,1))
original_msi = np.float32(original_msi) / 255

# max_patch, min_patch = np.max(original_pan, axis=(0,1)), np.min(original_pan, axis=(0,1))
original_pan = np.float32(original_pan) / 255

'''generating ms image with gaussian kernel'''
# sig = (1/(2*(2.772587)/4**2))**0.5
# kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
# new_lrhs = []
# for i in range(original_msi.shape[-1]):
#     temp = signal.convolve2d(original_msi[:,:, i], kernel, boundary='wrap',mode='same')
#     temp = np.expand_dims(temp, -1)
#     new_lrhs.append(temp)
# new_lrhs = np.concatenate(new_lrhs, axis=-1)
# used_ms = new_lrhs[0::4, 0::4, :]

#'''generating ms image with bicubic interpolation'''
used_ms = cv2.resize(original_msi, (original_msi.shape[1]//4, original_msi.shape[0]//4), cv2.INTER_CUBIC)

'''generating pan image with gaussian kernel'''
# used_pan = signal.convolve2d(original_pan, kernel, boundary='wrap',mode='same')
# used_pan = np.expand_dims(used_pan, -1)
# used_pan = used_pan[0::4, 0::4, :]

#'''generating pan image with vitual spectral kernel'''
#spectral_kernel = np.array([[0.1], [0.1], [0.4], [0.4]])
#used_pan = np.dot(original_msi, spectral_kernel)

#'''generating ms image with bicubic interpolation'''
# used_pan = cv2.resize(original_pan, (original_pan.shape[1]//4, original_pan.shape[0]//4), cv2.INTER_CUBIC)
used_pan = np.expand_dims(original_pan, -1)
print(used_ms)
gt = np.uint8(255*original_msi)

print('ms shape: ', used_ms.shape, 'pan shape: ', used_pan.shape)

'''setting save parameters'''
save_images = True
save_channels = [0, 1, 2] #BGR-NIR for GF2
save_dir='./results/'
# cv2.imwrite(save_dir+'ms.tif', np.uint8(255*used_ms)[:, :, save_channels])
# cv2.imwrite(save_dir+'pan.tif', np.uint8(255*used_pan)[:, :])
# cv2.imwrite(save_dir+'gt.tif', gt[:, :, save_channels])
if save_images and (not os.path.isdir(save_dir)):
    os.makedirs(save_dir)

'''evaluating all methods'''
ref_results={}
ref_results.update({'metrics: ':'  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q'})
no_ref_results={}
no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})

'''Bicubic method'''
print('evaluating Bicubic method')
fused_image = Bicubic(used_pan[:, :, :], used_ms[:, :, :])
print(gt.shape)
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'Bicubic    ':temp_ref_results})
no_ref_results.update({'Bicubic    ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'Bicubic.tif')
    # cv2.imwrite(save_dir+'Bicubic.tif', fused_image[:, :, save_channels])

'''Brovey method'''
print('evaluating Brovey method')
fused_image = Brovey(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'Brovey     ':temp_ref_results})
no_ref_results.update({'Brovey     ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'Brovey.tif')
    # cv2.imwrite(save_dir+'Brovey.tif', fused_image[:, :, save_channels])
    
'''PCA method'''
print('evaluating PCA method')
fused_image = PCA(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'PCA        ':temp_ref_results})
no_ref_results.update({'PCA        ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'PCA.tif')
    # cv2.imwrite(save_dir+'PCA.tif', fused_image[:, :, save_channels])
    
'''IHS method'''
print('evaluating IHS method')
fused_image = IHS(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'IHS        ':temp_ref_results})
no_ref_results.update({'IHS        ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'IHS.tif')
    # cv2.imwrite(save_dir+'IHS.tif', fused_image[:, :, save_channels])
    
'''SFIM method'''
print('evaluating SFIM method')
fused_image = SFIM(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'SFIM       ':temp_ref_results})
no_ref_results.update({'SFIM       ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'SFIM.tif')
    # cv2.imwrite(save_dir+'SFIM.tif', fused_image[:, :, save_channels])

'''GS method'''
print('evaluating GS method')
fused_image = GS(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GS         ':temp_ref_results})
no_ref_results.update({'GS         ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'GS.tif')
    # cv2.imwrite(save_dir+'GS.tif', fused_image[:, :, save_channels])
    
'''Wavelet method'''
print('evaluating Wavelet method')
fused_image = Wavelet(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'Wavelet    ':temp_ref_results})
no_ref_results.update({'Wavelet    ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'Wavelet.tif')
    # cv2.imwrite(save_dir+'Wavelet.tif', fused_image[:, :, save_channels])

'''MTF_GLP method'''
print('evaluating MTF_GLP method')
fused_image = MTF_GLP(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'MTF_GLP    ':temp_ref_results})
no_ref_results.update({'MTF_GLP    ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'MTF_GLP.tif')
    # cv2.imwrite(save_dir+'MTF_GLP.tif', fused_image[:, :, save_channels])

'''MTF_GLP_HPM method'''
print('evaluating MTF_GLP_HPM method')
fused_image = MTF_GLP_HPM(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'MTF_GLP_HPM':temp_ref_results})
no_ref_results.update({'MTF_GLP_HPM':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'MTF_GLP_HPM.tif')
    # cv2.imwrite(save_dir+'MTF_GLP_HPM.tif', fused_image[:, :, save_channels])

'''GSA method'''
print('evaluating GSA method')
fused_image = GSA(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GSA        ':temp_ref_results})
no_ref_results.update({'GSA        ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'GSA.tif')
    # cv2.imwrite(save_dir+'GSA.tif', fused_image[:, :, save_channels])

'''CNMF method'''
print('evaluating CNMF method')
fused_image = CNMF(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'CNMF       ':temp_ref_results})
no_ref_results.update({'CNMF       ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'CNMF.tif')
    # cv2.imwrite(save_dir+'CNMF.tif', fused_image[:, :, save_channels])

'''GFPCA method'''
print('evaluating GFPCA method')
fused_image = GFPCA(used_pan[:, :, :], used_ms[:, :, :])
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GFPCA      ':temp_ref_results})
no_ref_results.update({'GFPCA      ':temp_no_ref_results})
#save
if save_images:
    save_img = Image.fromarray(fused_image, 'CMYK')
    save_img.save(save_dir+'GFPCA.tif')
    # cv2.imwrite(save_dir+'GFPCA.tif', fused_image[:, :, save_channels])

# '''PNN method'''
# print('evaluating PNN method')
# fused_image = PNN(used_pan[:, :, :], used_ms[:, :, :])
# temp_ref_results = ref_evaluate(fused_image, gt)
# temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
# ref_results.update({'PNN        ':temp_ref_results})
# no_ref_results.update({'PNN        ':temp_no_ref_results})
# #save
# if save_images:
#     cv2.imwrite(save_dir+'PNN.tif', fused_image[:, :, save_channels])

# '''PanNet method'''
# print('evaluating PanNet method')
# fused_image = PanNet(used_pan[:, :, :], used_ms[:, :, :])
# temp_ref_results = ref_evaluate(fused_image, gt)
# temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
# ref_results.update({'PanNet     ':temp_ref_results})
# no_ref_results.update({'PanNet     ':temp_no_ref_results})
# #save
# if save_images:
#     cv2.imwrite(save_dir+'PanNet.tif', fused_image[:, :, save_channels])

# ''''print result'''
print('################## reference comparision #######################')
for index, i in enumerate(ref_results):
    if index == 0:
        print(i, ref_results[i])
    else:    
        print(i, [round(j, 4) for j in ref_results[i]])
print('################## reference comparision #######################')
      
      
print('################## no reference comparision ####################')
for index, i in enumerate(no_ref_results):
    if index == 0:
        print(i, no_ref_results[i])
    else:    
        print(i, [round(j, 4) for j in no_ref_results[i]])
print('################## no reference comparision ####################')


