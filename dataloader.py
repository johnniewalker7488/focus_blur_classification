import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torch.utils.data import Sampler

import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import get_mean_std


def loader_dist(loader):
    """ Shows distribution of class '1' in dataloader batches"""
    foc = []
    for idx, (x, y) in enumerate(loader):
        share = y.sum() / len(y)
        foc.append(share)
    plt.hist(np.array(foc), bins=20)
    plt.show()
    

data_dir = './FocusPath_Full/'

def get_datasets(crop_size=224, norm=False, mean=None, std=None):
    
    if norm:
        train_tfms = tt.Compose([
            tt.RandomCrop(crop_size),
            tt.ToTensor(),
            tt.Normalize(mean, std),
        ])

        valid_tfms = tt.Compose([
            tt.RandomCrop(crop_size),
            tt.ToTensor(),
            tt.Normalize(mean, std),
        ])
        
        test_tfms = tt.Compose([
            tt.CenterCrop(crop_size),
            tt.ToTensor(),
            tt.Normalize(mean, std),
        ])
    
    else:
        train_tfms = tt.Compose([
            tt.RandomCrop(crop_size),
            tt.ToTensor(),
        ])

        valid_tfms = tt.Compose([
            tt.RandomCrop(crop_size),
            tt.ToTensor(),
        ])
        
        test_tfms = tt.Compose([
            tt.CenterCrop(crop_size),
            tt.ToTensor(),
        ])

    train_ds = ImageFolder(data_dir + '/train', train_tfms)
    valid_ds = ImageFolder(data_dir + '/val', valid_tfms)
    test_ds = ImageFolder(data_dir + '/test', test_tfms)

    return train_ds, valid_ds, test_ds


def focus_blur_ds(dataset):
    focused_idx = []
    blur_idx = []
    for i in dataset.imgs:
        if i[1] == 1:
            focused_idx.append(dataset.imgs.index(i))
        else:
            blur_idx.append(dataset.imgs.index(i))
    focused_ds = torch.utils.data.Subset(dataset, focused_idx)
    blur_ds = torch.utils.data.Subset(dataset, blur_idx)
    return focused_ds, blur_ds


def get_loader(root_dir, dataset, batch_size=64):
    dataset = dataset
    dataset_size = len(dataset)
    
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        class_size = len(files)
        if class_size > 0:
            class_weights.append(class_size/dataset_size)
            
    sample_weights = [0] * len(dataset)
    loop = tqdm(enumerate(dataset), total=len(dataset), leave=True)
    
    for idx, (image, label) in loop:
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
        loop.set_description('Mapping sample weights')
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=28, pin_memory=True)
    
    return loader


def show_batch(loader, figsize=16):
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(loader.batch_size)))
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
