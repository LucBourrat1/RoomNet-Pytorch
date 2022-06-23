from torch.utils.data import Dataset
from torchvision.transforms import transforms
from glob import glob
import os
import numpy as np
from PIL import Image
import torch

class LSUN_dset(Dataset):
    def __init__(self, data_root, transform, train=True):
        super(LSUN_dset, self).__init__()
        self.root = data_root
        self.train = train
        self.transform = transform
        
        self.data_list = glob(os.path.join(data_root, 'flip0', '*.npz'))
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        
        if self.train:
            # Random Flip
            choice = np.random.choice([0,1], 1)[0]
            if choice == 1:
                data_path = data_path.replace('flip0', 'flip1')
            
            data = dict(np.load(data_path))

            img = data['im']
            img = self.transform(img)
            
            lay = torch.tensor(data['lay']).float()
            label = torch.tensor(data['label']).long()
            mask_f = torch.tensor(data['mask_forward']).float()
            mask_b = torch.tensor(data['mask_backward']).float()
            return img, lay, mask_f, mask_b, label
            
        else:
            data_path_L = data_path
            data_path_R = data_path.replace('flip0', 'flip1')
            
            data_L = dict(np.load(data_path_L))
            data_R = dict(np.load(data_path_R))
            
            img_L = data_L['im']            
            img_L = self.transform(img_L)
            
            img_R = data_R['im']            
            img_R = self.transform(img_R)
            
            lay_L = torch.tensor(data_L['lay']).float()
            lay_R = torch.tensor(data_R['lay']).float()
            
            label = torch.tensor(data_L['label']).long()
            
            mask_L_f = torch.tensor(data_L['mask_forward']).float()
            mask_L_b = torch.tensor(data_L['mask_backward']).float()
            
            mask_R_f = torch.tensor(data_R['mask_forward']).float()
            mask_R_b = torch.tensor(data_R['mask_backward']).float()
            
            return img_L, lay_L, mask_L_f, mask_L_b, img_R, lay_R, mask_R_f, mask_R_b, label


def load_LSUN(args):
    tr_root = os.path.join(args.data_root, 'train')
    val_root = os.path.join(args.data_root, 'validation')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    tr_dataset = LSUN_dset(tr_root, transform, train=True)
    val_dataset = LSUN_dset(val_root, transform, train=False)
    return tr_dataset, val_dataset

    
if __name__=='__main__':
    root = '/home/dataset/LSUN/processed/train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dset = LSUN_dset(root, transform, train=False)
    dset.__getitem__(1)