import torch
from time import time
from utils import accuracy
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_net(model, test_loader):
    test_accs = []
    
    start = time()
#     print('Running on test set...')
    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    
    for batch_idx, (test_data, test_target) in loop:
        model.eval()  

        test_data = test_data.to(device)
        test_target = test_target.to(device)  
        test_pred = model(test_data)  
        test_acc = accuracy(test_pred, test_target)  
        test_accs.append(test_acc.item())
        loop.set_description('Running on test set')
        
    avg_test_acc = torch.tensor(test_accs).mean().item()
    end_time = time() - start
    print(f'Accuracy on test: {round(avg_test_acc, 4)}\t Time: {round(end_time, 2)}')
    
    return test_accs