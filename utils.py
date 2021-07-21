import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt

import os
from tqdm import tqdm
from time import time

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def show_img(img_path):
    
    """Shows an image, normalized image and their distributions
    Args:
        img_path: path to the image to be shown
    """  
    img = Image.open(img_path)
    img_np = np.array(img)
    transform = tt.Compose([tt.ToTensor()])
    img_tr = transform(img)
    mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])
    transform_norm = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])
    img_norm = transform_norm(img)
    img_norm = np.transpose(img_norm, (1, 2, 0))
    img_np_norm = np.array(img_norm)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 7))
    ax1.imshow(img)
    ax2.hist(img_np.ravel(), bins=50, density=True);
    ax3.imshow(img_norm)
    ax4.hist(img_np_norm.ravel(), bins=50, density=True)
    plt.show()
     
        
def get_mean_std():
    data_dir = './FocusPath_Full/'
    transforms = tt.Compose([tt.ToTensor()])
    mean_std_ds = ImageFolder(os.path.join(data_dir, 'train'), transforms)
    mean_std_loader = DataLoader(mean_std_ds, batch_size=64, num_workers=28)
    chan_sum, chan_squared_sum, num_batches = 0, 0, 0
    loop = tqdm(mean_std_loader, total=len(mean_std_loader), leave=True)
    
    for data, _ in loop:
        chan_sum += torch.mean(data, dim=[0, 2, 3])
        chan_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
        loop.set_description('Calculating mean and std')
        
    mean = chan_sum / num_batches
    std = (chan_squared_sum / num_batches)**0.5
    
    return mean, std
        

def test_model(model, num_examples=64, num_classes=2, input_size=224, input_channels=3):
    
    x = torch.randn((num_examples, input_channels, input_size, input_size))
    model = model(input_channels, num_classes)
    start = time()
    print(f'Random batch of size {num_examples} output shape:', model(x).shape)
    end = time() - start
    print(f'Forward pass ran in {round(end, 2)} seconds')


def count_parameters(model):
    '''
    counts number of NN parameters
    '''
    tot = 0
    for n, p in model.named_parameters():
        tot += p.numel()
        print(n, p.numel())
    print(f'Total: {tot}')

    
def get_lr(optimizer):
    """ Returns current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def save_checkpoint(state, file_name='model_weights.pth.tar'):
    #     print('--> Saving checkpoint')
    torch.save(state, file_name)


def load_checkpoint(checkpoint, optimizer):
    print('--> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def plot_losses(history, learning_rates, train_loader, file_name='loss', figsize=(10, 7)):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    lrs = [x for i, x in enumerate(learning_rates) if i % (len(train_loader)) == 0]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')

    min_val_loss = val_losses.index(min(val_losses)) + 1
    ax1.axvline(min_val_loss, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('learning rate')
    ax2.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='green', linestyle='--')

    ax1.grid(True)
    fig.legend()
    fig.tight_layout()
#     print('Saving losses plot')
#     plt.savefig(f'./learning_curves/{file_name}')
    plt.show()


def plot_accuracies(history, learning_rates, train_loader, file_name='accs', figsize=(10, 7)):
    train_accs = [x['train_acc'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    lrs = [x for i, x in enumerate(learning_rates) if i % (len(train_loader)) == 0]

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accs')
    ax1.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy')
    ax1.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('learning rate')
    ax2.plot(range(1, len(lrs) + 1), lrs, label='Learning Rate', color='green', linestyle='--')

    ax1.grid(True)
    fig.legend()
    fig.tight_layout()
#     print('Saving accuracies plot')
#     plt.savefig(f'./learning_curves/{file_name}')
    plt.show()
    
    
def predict_image(train_set, img_path, model, mean, std, crop_size=224):
    image = Image.open(img_path)
    transform = tt.Compose([tt.CenterCrop(crop_size), 
                            tt.ToTensor(),
                            tt.Normalize(mean, std)])
    x = transform(image)
    x = x.unsqueeze(0).to(device)
    y = model(x)
    _, pred = torch.max(y, dim=1)
    return train_set.classes[pred[0].item()]