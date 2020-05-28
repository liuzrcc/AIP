import os
import sys
import time
import argparse
import numpy as np
import random
from tqdm import tqdm

sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
device = 'cuda:0'

import recsys_models
import config

data_train = 'amazon'
if data_train == 'amazon':

    dataset_name = 'AmazonMenWithImgPartitioned.npy'

    dataset = np.load('../data/'+ dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('../data/amazon_one_k_cold.npy')

elif data_train == 'tradesy':
    
    dataset_name = 'TradesyImgPartitioned.npy'
    dataset = np.load('../data/' + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('../data/tradesy_one_k_cold.npy')

def metrics_hr(model, val_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j in val_loader:
        user = user.to(device)
        item_i = item_i.to(device)
        item_j = item_j.to(device)

        prediction_i, prediction_j = model(user, item_i, item_j)
        to_rank = np.append(prediction_j.detach().cpu().numpy(), prediction_i[0].detach().cpu().numpy())
        _, indices = torch.topk(torch.tensor(to_rank), top_k)
        if 100 in indices:
            HR.append(1)
        else:
            HR.append(0)
    return np.mean(HR)

class trainset(Dataset):
    def __init__(self):
        self.target = train_ls

    def __getitem__(self, index):
        target = self.target[index]
        return target[0], target[1], target[2]

    def __len__(self):
        return len(self.target)

class testset(Dataset):
    def __init__(self):
        self.target = test_ls

    def __getitem__(self, index):
        target = self.target[index]
        return target[0], target[1], target[2]

    def __len__(self):
        return len(self.target)
    
class valset(Dataset):
    def __init__(self):
        self.target = val_ls

    def __getitem__(self, index):
        target = self.target[index]
        return target[0], target[1], target[2]

    def __len__(self):
        return len(self.target)

oneiteration = 0
for item in user_train: oneiteration+=len(user_train[item])
    

def sample(user):
    u = random.randrange(usernum)
    numu = len(user[u])
    i = user[u][random.randrange(numu)][b'productid']
    M=set()
    for item in user[u]:
        M.add(item[b'productid'])
    while True:
        j=random.randrange(itemnum)
        if (not j in M) and (not j in cold_list): break
    return (u,i,j)

def sample_val(u_idx):
    u = u_idx
    user = user_validation[u_idx]
    i = user[random.randrange(1)][b'productid']
    M=set()
    for item in user:
        M.add(item[b'productid'])
    while True:
        j=random.randrange(itemnum)
        if (not j in M) and (not j in cold_list): break
    return list((u,i,j))


val_ls = [list(sample_val(u_idx) for i in range(100)) for u_idx in tqdm(range(usernum))]
val_ls = np.array(val_ls).reshape(-1, 3)

val_data  = valset()
val_loader = DataLoader(val_data, batch_size = 100, 
                       shuffle = False, num_workers = 4)


model = recsys_models.BPR(usernum, itemnum, 64)
model.to(device)


if data_train == 'amazon':
    # for amazon
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
elif data_train == 'tradesy':
    # for tradesy
    optimizer = optim.SGD(model.parameters(), lr=0.5, weight_decay=0.001)


writer = SummaryWriter()
writer_ct = 0

count, best_hr = 0, 0
for epoch in tqdm(range(2000)):
    model.train() 
    start_time = time.time()
    train_ls = [list(sample(user_train)) for _ in range(oneiteration)]

    train_data  = trainset()

    for data in DataLoader(train_data, batch_size = 4096, 
                       shuffle = True, pin_memory = True, num_workers = 6):
        
        user, item_i, item_j = data

        model.zero_grad()
        prediction_i, prediction_j = model(user.to(device), item_i.to(device), item_j.to(device))
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        loss.backward()
        optimizer.step()
        writer.add_scalar('runs/loss', loss.item(), count)
        count += 1000
    model.eval()
    
    HR = metrics_hr(model, val_loader, 5)
    writer.add_scalar('runs/HR_at5', HR, count)
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}".format(np.mean(HR)))

    if HR > best_hr:
        best_hr, best_epoch = HR, epoch
        if True:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model, '{}amazon_BPR_trial.pt'.format(config.model_path))

    print("End. Best epoch {:03d}: HR = {:.3f}".format(best_epoch, best_hr,))