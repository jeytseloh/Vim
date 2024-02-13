import os
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset
import torch

from vim_3d.utils.utils_3d import adjust_window, zscore_normalization, ct_normalization, calculate_foreground_stats
from vim_3d.utils.Data_Transform import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose,Resize

class Dataset_3D(Dataset):
    """"
    DatasetFolder/
    ├── imagesTr
    ├── labelsTr
    ├── imagesTs
    ├── labelsTs

    Images and its labels assumed to have same filename.
    """
    def __init__(self, dataset_path, is_train=True, transform=None):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.tr_flag = "Tr"
            self.img_path = os.path.join(dataset_path, f"images{self.tr_flag}/")
            self.label_path = os.path.join(dataset_path, f"labels{self.tr_flag}/")
        # else:
        #     self.ts_flag = "Ts"
        #     self.img_path = os.path.join(dataset_path, f"images{self.ts_flag}/")
        #     self.label_path = os.path.join(dataset_path, f"labels{self.ts_flag}/")

        self.img_list = [f for f in os.listdir(self.img_path) if not f.startswith('._')] # macos
        # self.img_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_filename = self.img_list[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.img_path + img_filename))
        label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_path + img_filename)).astype(np.int8)
        # expand dims -> (1, D, H, W)
        img = torch.FloatTensor(img).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        img = ct_normalization(img, p995=255., p005=22.4, mean=41.43, std=30.84)

        return img, label, img_filename
    
def build_dataset_3d(is_train, args):
    transforms = build_transforms_3d(is_train, args)
    dataset = Dataset_3D(dataset_path=args.data_path, is_train=is_train, transform=transforms)
    return dataset

def build_transforms_3d(is_train, args):
    if is_train:
        transforms = Compose([
            RandomFlip_LR(prob=0.5),
            RandomFlip_UD(prob=0.5),
            # RandomRotate()
        ])
    return transforms



if __name__ == '__main__':
    dataset_path = "/Volumes/Expansion/VimExperiments/ATM22TestDataset/"
    train_data = Dataset_3D(dataset_path)

    for img, label, fid in train_data:
        print(torch.max(img), torch.min(img))

