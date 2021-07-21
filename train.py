import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from time import time

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_net(model, num_epochs, train_loader, val_loader, max_lr=0.01, cycles=1):

    optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs*len(train_loader))//cycles)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs,
#                                                     steps_per_epoch=len(train_loader))

    history = []
    learning_rates = []
    total_val_loss = []

    for epoch in range(num_epochs):
        train_losses = []
        train_accs = []
        lrs = []
        start = time()
        
#         loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for batch_idx, (train_data, train_target) in enumerate(train_loader):
            # allocate batch tensors to device
            train_data = train_data.to(device)
            train_target = train_target.to(device)

            model.zero_grad()  # zero gradients for next calculation

            train_pred = model(train_data)  # forward pass

            train_loss = F.cross_entropy(train_pred, train_target)  # calculate loss per batch
            train_losses.append(train_loss.item())

            train_acc = utils.accuracy(train_pred, train_target)  # calculate accuracy per batch
            train_accs.append(train_acc.item())

            train_loss.backward()  # calculate gradients with backprop
            optimizer.step()  # update weights
            scheduler.step()
            lrs.append(utils.get_lr(optimizer))
#             loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        # validation step
        val_accs = []  # accuracies for every batch in validation set
        val_losses = []  # losses for every batch in validation set

        for batch_idx, (val_data, val_target) in enumerate(val_loader):
            model.eval()  # enable no_grad for evaluation

            val_data = val_data.to(device)  # allocate batch tensor to device
            val_target = val_target.to(device)  # allocate batch tensor to device

            val_pred = model(val_data)  # forward pass for batch in validation set

            val_loss = F.cross_entropy(val_pred, val_target)  # calculate loss per batch
            val_losses.append(val_loss.item())

            val_acc = utils.accuracy(val_pred, val_target)  # calculate accuracy per batch
            val_accs.append(val_acc.item())

        avg_train_loss = torch.tensor(train_losses).mean().item()  # train loss per epoch
        avg_train_acc = torch.tensor(train_accs).mean().item()  # train accuracy per epoch
        avg_val_loss = torch.tensor(val_losses).mean().item()  # validation loss per epoch
        avg_val_acc = torch.tensor(val_accs).mean().item()  # validation accuracy per epoch

        total_val_loss.append(avg_val_loss)

        end_time = time() - start
        line = "Epoch[{:d}] Train_loss: {:.4f}  Val_loss: {:.4f}\t Train_acc: {:.4f}  Val_acc: {:.4f} LR: {:.5f} time: {:.2f}".format(
            epoch,
            round(train_loss.item(), 4),
            round(avg_val_loss, 4),
            round(avg_train_acc, 4),
            round(avg_val_acc, 4),
            utils.get_lr(optimizer),
            end_time)
        print(line)

        if avg_val_loss <= torch.tensor(total_val_loss).min().item():
#             checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
#             utils.save_checkpoint(checkpoint)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './model_weights/model_weights.pth')

        # dict with metrics per epoch for further visualization
        epoch_end = {'train_loss': avg_train_loss, 'train_acc': avg_train_acc,
                     'val_loss': avg_val_loss, 'val_acc': avg_val_acc}
        history.append(epoch_end)
        learning_rates.append(lrs)

    print(f'Max validation accuracy: {round(torch.tensor([x["val_acc"] for x in history]).max().item(), 4)}')
    learning_rates = torch.tensor(learning_rates).clone().detach().flatten()

    return history, learning_rates