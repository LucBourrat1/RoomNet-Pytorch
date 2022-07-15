# Reference : https://github.com/GitBoSun/roomnet/blob/master/roomnet/prepare_data.py
import os
import numpy as np
import cv2
import scipy.io as sio
import argparse
from tqdm import tqdm

def guassian_2d(x_mean, y_mean, dev=5.0):
	x, y = np.meshgrid(np.arange(out_s), np.arange(out_s))
	#z=(1.0/(2.0*np.pi*dev*dev))*np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
	z=np.exp(-((x-x_mean)**2+ (y-y_mean)**2)/(2.0*dev**2))
	return z

  
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_type', type=str, default='train')
    args = parser.parse_args()
 
    if args.train_type == 'train':
        TRAIN = True
    else:
        TRAIN = False
    
    if TRAIN:
        im_path='/home/luc/drive/datasets/lsun/train/image'
        mat='/home/luc/drive/datasets/lsun/training.mat'
        outpath='/home/luc/Dev/RoomNet-Pytorch/data/processed/train'
        stage = 'training'
    
    else:
        im_path='/home/luc/drive/datasets/lsun/val/image'
        mat='/home/luc/drive/datasets/lsun/validation.mat'
        outpath='/home/luc/Dev/RoomNet-Pytorch/data/processed/validation'
        stage = 'validation'
        
        
    s=320
    out_s=40
    l_list=[0,8,14,20,24,28,34,38,42,44,46]
    
    flip_idx={}
    flip_idx[0]=[7,8,5,6,3,4,1,2]
    flip_idx[1]=[4,5,6,1,2,3]
    flip_idx[2]=[4,5,6,1,2,3]
    flip_idx[3]=[1,4,3,2]
    flip_idx[4]=[1,4,3,2]
    flip_idx[5]=[1,3,2,4,6,5]
    flip_idx[6]=[2,1,4,3]
    flip_idx[7]=[3,4,1,2]
    flip_idx[8]=[2,1]
    flip_idx[9]=[2,1]
    flip_idx[10]=[1,2]
    
    os.makedirs(os.path.join(outpath, 'flip0'), exist_ok=True)
    os.makedirs(os.path.join(outpath, 'flip1'), exist_ok=True)
    
    data = sio.loadmat(mat)
    data=data[stage][0]
    idx = 0
    for item in tqdm(data):
        idx += 1
        name=item[0][0]
        ltype=item[2][0][0]
        pts=item[3]
        
        h,w=item[4][0]
        #im=cv2.imread(os.path.join(im_path, name+'.jpg'))
        im=cv2.imread(os.path.join(im_path, f"{str(idx).zfill(4)}.jpg"))
        im=cv2.resize(im, (s,s), interpolation = cv2.INTER_CUBIC)
        
        class_label = ltype
        
        layout=np.zeros((48, out_s, out_s))
        mask_forward = np.zeros((48, out_s, out_s))
        mask_backward = np.zeros((48, out_s, out_s))
        
        for i, pt in enumerate(pts):
            x_mean=int(pt[0]*(40.0/w))
            y_mean=int(pt[1]*(40.0/h))
            
            if x_mean==40:
                x_mean=39
            if y_mean==40:
                y_mean=39

            gaussian_pts = guassian_2d(x_mean, y_mean)
            layout[l_list[ltype]+flip_idx[ltype][i]-1, :, :]= gaussian_pts
            mask_forward[l_list[ltype]+flip_idx[ltype][i]-1, :, :] = (gaussian_pts > 0.7).astype('float')
            mask_backward[l_list[ltype]+flip_idx[ltype][i]-1, :, :] = (gaussian_pts < 0.7).astype('float')
            
        np.savez(os.path.join(outpath, 'flip0', '%s.npz'%(name)), im=im, lay=layout, label=class_label, mask_forward=mask_forward, mask_backward=mask_backward)
  	
        im = cv2.flip(im, 1)
        
        layout=np.zeros((48, out_s, out_s))
        mask_forward = np.zeros((48, out_s, out_s))
        mask_backward = np.zeros((48, out_s, out_s))
        
        for i, pt in enumerate(pts):
            x_mean = int(pt[0] * (40.0 / w))
            y_mean = int(pt[1] * (40.0 / h))
            if x_mean == 40:
                x_mean = 39
            if y_mean == 40:
                y_mean = 39
            
            x_mean = 39 - x_mean 
            
            gaussian_pts = guassian_2d(x_mean, y_mean)
            layout[l_list[ltype]+flip_idx[ltype][i]-1, :, :]= gaussian_pts
            mask_forward[l_list[ltype]+flip_idx[ltype][i]-1, :, :] = (gaussian_pts > 0.7).astype('float')
            mask_backward[l_list[ltype]+flip_idx[ltype][i]-1, :, :] = (gaussian_pts < 0.7).astype('float')
            
        np.savez(os.path.join(outpath, 'flip1', '%s.npz' % (name)), im=im, lay=layout, label=class_label,  mask_forward=mask_forward, mask_backward=mask_backward)