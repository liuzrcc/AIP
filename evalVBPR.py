import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO, BytesIO
import threading
from tqdm import tqdm_notebook, tqdm

from recsys_models import pthDVBPR, pthVBPR, Normalize
# from recsys_attacks import INSA_DVBPR, EXPA_DVBPR, INSA_VBPR, Alex_EXPA, INSA_AlexRank
from utils import cos_distance

device = 'cuda:0'

data_for_experiment = 'amazon'

data_root = '/workspace/zhuoran/data/'
if data_for_experiment == 'amazon':
    dataset_name = 'AmazonMenWithImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

    alex_4096_cnn_f = np.load('/workspace/zhuoran/data/ama_alexnet_features.npy')
elif data_for_experiment == 'tradesy':
    dataset_name = 'TradesyImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    
    alex_4096_cnn_f = np.load('/workspace/zhuoran/data/tradesy_alexnet_features.npy')
    
    
model = torch.load('./models/'+ data_for_experiment + '_k100_VBPR.pt', map_location = device)
model.eval()

for param in model.parameters():
    param.requires_grad = False

    
VBPR_UI_m_BPR = np.load('./intermediate_score_index/VBPR_'+ data_for_experiment + '_k100.npy')

new_model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(new_model.classifier.children())[:-1])
new_model.classifier = new_classifier
new_model.eval().to(device)

VBPR_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

def default_loader(path):
    img_pil =  Image.open((path)).convert('RGB')
    img_tensor = orginal_transform(img_pil)
    return img_tensor

class testset_img(Dataset):
    def __init__(self, file_item_i, loader=default_loader):
        self.images_i = file_item_i
        self.loader = loader

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        return img_i

    def __len__(self):
        return len(self.images_i)

cold_list = np.load('./data/'+ data_for_experiment + '_one_k_cold.npy')
bpr_score = np.load('./intermediate_score_index/VBPR_' + data_for_experiment + '_k100.npy')



eval_root = './adv_output/VBPR/' + item + '/'
def default_loader(path):
    img_pil =  Image.open((path)).convert('RGB')
    img_tensor = VBPR_test_transform(img_pil)
    return img_tensor

class testset_img(Dataset):
    def __init__(self, file_item_i, loader=default_loader):
        self.images_i = file_item_i
        self.loader = loader

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        return img_i

    def __len__(self):
        return len(self.images_i)

lslsls = []
for img in os.listdir(eval_root):
    if (img.split('.')[-1] == 'jpg') or (img.split('.')[-1] == 'png'):
        lslsls.append(img)

file_item_i = [eval_root + i  for i in lslsls]
item_data  = testset_img(file_item_i)

I = np.array([])
for data in tqdm_notebook(DataLoader(item_data, batch_size = 128, num_workers = 4)):
    if len(I) == 0:
        I = new_model(data.to(device)).cpu().data.numpy()
    else:
        I = np.append(I, new_model(data.to(device)).cpu().data.numpy(), axis = 0)

j_list = np.array([int(img.split('/')[-1].split('.')[0]) for img in file_item_i])

scores_temp = np.zeros((usernum, len(j_list)))

scores_temp = np.zeros((usernum, len(I)))
for i in tqdm_notebook(range(usernum)):
    x1 = torch.LongTensor(np.repeat(i, len(I)).astype(np.int32))
    x2 = torch.LongTensor(np.array(j_list).astype(np.int32))
    x3 = torch.tensor(I)
    res = model(x1.to(device), x2.to(device), x3.to(device))
    scores_temp[i] = res.detach().cpu().numpy()[0]
max_item_num = 100

HR_org_top5_count = []
HR_org_top10_count = []
HR_org_top20_count = []

HR_adv_top5_count = []
HR_adv_top10_count = []
HR_adv_top20_count = []

for i in tqdm_notebook(range(max_item_num)):

    to_test = np.append(bpr_score, scores_temp[:, i].reshape(-1, 1), axis = 1)
    rank_res = np.argsort(-1 * to_test)

    HR_org_top5_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :5]]).mean()])
    HR_org_top10_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :10]]).mean()])
    HR_org_top20_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :20]]).mean()])

    HR_adv_top5_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :5]]).mean()])
    HR_adv_top10_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :10]]).mean()])
    HR_adv_top20_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :20]]).mean()])

print('------------------------------')
print(np.array(HR_org_top5_count).mean())
print(np.array(HR_org_top10_count).mean())
print(np.array(HR_org_top20_count).mean())
print(np.array(HR_adv_top5_count).mean())
print(np.array(HR_adv_top10_count).mean())
print(np.array(HR_adv_top20_count).mean())
print('------------------------------')
np.save('./results/' + 'VBPR_' + item + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
    [np.array(HR_org_top10_count).mean()],
    [np.array(HR_org_top20_count).mean()],
    [np.array(HR_adv_top5_count).mean()],
    [np.array(HR_adv_top10_count).mean()],
    [np.array(HR_adv_top20_count).mean()]]))