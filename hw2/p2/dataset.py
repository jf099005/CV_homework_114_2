# ============================================================================
# File: dataset.py
# Date: 2026-03-27
# Author: TA
# Description: Dataset and DataLoader.
# ============================================================================

import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import ConcatDataset



def get_dataloader(
        dataset_dir,
        batch_size: int = 1,
        split: str = 'test',
        unlabel_annotation_path = None,
        return_img_names = False):
    '''
    Build a dataloader for given dataset and batch size.
    - Args:
        - dataset_dir: str, path to the dataset directory
        - batch_size: int, batch size for dataloader
        - split: str, 'train', 'val', or 'test'
    - Returns:
        - dataloader: torch.utils.data.DataLoader
    '''
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomCrop(32, padding=4),              # 隨機裁切 + padding
            transforms.RandomHorizontalFlip(p=0.5),            # 隨機水平翻轉
            transforms.RandomRotation(15),                     # 小角度旋轉

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = CIFAR10Dataset(
        dataset_dir, split=split,
        transform=transform, 
        unlabel_annotation_path = unlabel_annotation_path,
        return_img_names = return_img_names
    )
    if dataset[0] is None:
        raise NotImplementedError('No data found, check dataset.py and implement __getitem__() in CIFAR10Dataset class!')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(split=='train'),
                            num_workers=0,
                            pin_memory=True, 
                            drop_last=(split=='train'))
    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None, unlabel_annotation_path = None, return_img_names = False):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.return_img_names = return_img_names

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)

        if unlabel_annotation_path:
            assert split != 'test', f"parameter \"unlabel_annotation_path\" is given but the dataset split is {split}"
            with open(unlabel_annotation_path, 'r') as f:
                json_data = json.load(f)

        self.image_names = json_data['filenames']

        if self.split != 'test':
            self.labels = json_data['labels']
            assert len(self.labels) == len(self.image_names)
        
        if unlabel_annotation_path:
            self.image_paths = json_data['filepaths']
        else:
            self.image_paths = [
                os.path.join( self.dataset_dir, img_name )
                for img_name in self.image_names
            ]

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################
        img_path = self.image_paths[index] #os.path.join(self.dataset_dir, self.image_names[index])
        image = Image.open(img_path)
        image = self.transform(image)
        if self.split == 'test':
            if self.return_img_names:
                return {
                    'image_names': self.image_names[index],
                    'images': image, 
                }

            return {
                'images': image, 
            }
        
        label = self.labels[index]
        ###################### TODO End ########################            

        return {
            'images': image, 
            'labels': label
        }
    
def concat_loaders(base_dataloader:DataLoader, extra_dataloader:DataLoader):
    combined_dataset = ConcatDataset([base_dataloader.dataset, extra_dataloader.dataset])
    combined_loader = DataLoader(combined_dataset, ...)
    return combined_loader
