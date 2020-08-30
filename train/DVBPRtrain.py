import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO, BytesIO
import threading
from tqdm import tqdm
import argparse

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

import model as recsys_models


parser = argparse.ArgumentParser(description = "DVBPR train")
parser.add_argument("-data_set", "--data_train", help="Training data to use", default="amazon")
parser.add_argument("-gpu_id", "--gpu", type=int, help="Using GPU or not, cpu please use -1", default='0')
parser.add_argument("-factor_num", "--K", type=int, help="Length of latent factors.", default="100")
parser.add_argument("-epoch", "--training_epoch", type=int, help="Training epoches.", default="20")
parser.add_argument("-batch_size", "--batch_size", type=int, help="Training batch size.", default="128")
parser.add_argument("-lambda1", "--lambda1", type=float, help="Weight of regulizer for user embeddings.", default="1e-3")
parser.add_argument("-lambda2", "--lambda2", type=int, help="Weight of regulizer for network.", default="1")
parser.add_argument("-learning_rate", "--lr", type=float, help="Weight of regulizer for network.", default="1e-4")
parser.add_argument("-num_workers", "--numofworkers", type=int, help="Number of co-workers for dataloader.", default="4")
args = parser.parse_args()


# data_train = 'amazon'
# K = 100 # Latent dimensionality
# lambda1 = 0.001 # Weight decay
# lambda2 = 1.0 # Regularizer wfor theta_u
# learning_rate = 1e-4
# training_epoch = 20
# batch_size = 128
# dropout = 0.5 # Dropout, probability to keep units
# numofworkers=4 # number of workers for pytorch dataloader
# training_epoch = 20
# device = 'cuda:0'
if args.gpu == 0:
    device = 'cuda:0'
elif args.gpu == -1:
    device = 'cpu'

data_train = args.data_train
K = args.K # Latent dimensionality
lambda1 = args.lambda1 # Weight decay
lambda2 = args.lambda2 # Regularizer wfor theta_u
learning_rate = args.lr
training_epoch = args.training_epoch
batch_size = args.batch_size
dropout = 0.5 # Dropout, probability to keep units
numofworkers=args.numofworkers # number of workers for pytorch dataloader
training_epoch = args.training_epoch


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



def default_loader(path):
    img_pil =  Image.open(BytesIO(path)).convert('RGB')
    img_tensor = input_transform(img_pil)
    return img_tensor


class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images_i = file_train_i
        self.images_j = file_train_j
        self.target = train_ls
        self.loader = loader

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        fn_j = self.images_j[index]
        img_j = self.loader(fn_j)
        target = self.target[index]
        return img_i, img_j, target[0], target[1], target[2]

    def __len__(self):
        return len(self.images_i)


class testset(Dataset):
    def __init__(self, loader=default_loader):
        self.images_i = file_item_i
        self.loader = loader

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        return img_i

    def __len__(self):
        return len(self.images_i)


def AUC(train,test,U,I):
    ans=0
    cc=0
    for u in tqdm_notebook(train):
        i=test[u][0][b'productid']
        T=np.dot(U[u,:],I.T)
        cc+=1
        M=set()
        for item in train[u]:
            M.add(item[b'productid'])
        M.add(i)

        count=0
        tmpans=0
        for j in range(itemnum):
#         for j in random.sample(range(itemnum),100): #sample
            if j in M: continue
            if T[i]>T[j]: tmpans+=1
            count+=1
        tmpans/=float(count)
        ans+=tmpans
    ans/=float(cc)
    return ans

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

input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))])
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

model = recsys_models.pthDVBPR(K).to(device)
model.apply(init_weights)

thetau = recsys_models.User_Em(usernum, K).to(device)


file_item_i = [Item[i][b'imgs'] for i in range(itemnum)]

def Evaluate(thetau, model):

    for item in thetau.parameters():
        U_np = item.cpu().data.numpy()

    model_t = model.eval()

#     file_item_i = [Item[i][b'imgs'] for i in range(itemnum)]
    item_data  = testset()
#     change to tensor does not help to accelerate!!!!
    I = np.array([])
    for data in DataLoader(item_data, batch_size = batch_size, num_workers = numofworkers):
        if len(I) == 0:
            I = model(data.to(device)).cpu().data.numpy()
        else:
            I = np.append(I, model(data.to(device)).cpu().data.numpy(), axis = 0)

    return AUC(user_train, user_validation, U_np, I)


oneiteration = 0
for item in user_train: oneiteration+=len(user_train[item])

writer = SummaryWriter()
writer_ct = 0


params = list(thetau.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

# scheduler = MultiStepLR(optimizer, milestones=[55,58], gamma=0.1)

for epoch in tqdm(range(training_epoch)):

    batch_count = 0
#      bootstrap
    train_ls = [list(sample(user_train)) for _ in range(oneiteration)]

    file_train_i = [Item[i][b'imgs'] for _,i,j in train_ls]
    file_train_j = [Item[j][b'imgs'] for _,i,j in train_ls]

    train_data  = trainset()

    for data in DataLoader(train_data, batch_size = batch_size,
                           shuffle = False, pin_memory = True, num_workers = numofworkers):

        nn_l2_reg = torch.tensor(0.).to(device)
        for name, param in model.named_parameters():
            if name.split('.')[-1] == 'weight':
                nn_l2_reg += torch.sum((param) **2 ) / 2

        batch_image1, batch_image2, u, _, _ = data

        result1 = model(batch_image1.to(device))
        result2 = model(batch_image2.to(device))

        cost_train = torch.log(torch.sigmoid(torch.mul(thetau(u),torch.sub(result1,result2)).sum(1))).sum()
        l2_reg = torch.sum((thetau(u)) **2 ) / 2

        loss = -1 * (cost_train - lambda1 * nn_l2_reg - lambda2 * l2_reg)


        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

#         for param in thetau.parameters():
#             print('User embed norm is', torch.sum((param) **2 ) / 2)

        batch_count += batch_size

        if (batch_count % (batch_size * 10)) == 0:

            writer_ct += 1
            n_iter = writer_ct
            writer.add_scalar('Loss/cost_train', cost_train, n_iter)
            writer.add_scalar('Loss/l2_reg', l2_reg, n_iter)

    auc_val = Evaluate(thetau, model)
    writer.add_scalar('Loss/val_AUC', auc_val, epoch)

#         scheduler.step()
    for item in thetau.parameters():
        U_np = item.cpu().data.numpy()

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'U': U_np
            }, '../models/ckpt/' + data_train + '_K' + K + '_' + str(epoch) + '.tar')
