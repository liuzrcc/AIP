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

data_for_experiment = 'tradesy'

data_root = './data/'
if data_for_experiment == 'amazon':
    dataset_name = 'AmazonMenWithImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

    alex_4096_cnn_f = np.load('./data/ama_alexnet_features.npy')
elif data_for_experiment == 'tradesy':
    dataset_name = 'TradesyImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    
    alex_4096_cnn_f = np.load('./data/tradesy_alexnet_features.npy')
    
model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
model.eval().to(device)

alexrank_test_transform = transforms.Compose([
    transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

for param in model.parameters():
    param.requires_grad = False
    
item_dict = {}
for u in tqdm_notebook(range(usernum)):
    for j in user_train[u]:
        item_id = j[b'productid']
        if u not in item_dict:
            item_dict[u] = [item_id]
        else:
            item_dict[u].append(item_id) 
            
bpr_scores = np.load('./intermediate_score_index/'+ data_for_experiment + '_visrank_bpr_scores_alex.npy')


eval_root = './adv_output/VISRANK/' + item + '/'

test_feature = np.empty((0, 4096))
for img in tqdm_notebook(os.listdir(eval_root)):
    if img.split('.')[-1] == 'png':
        test_feature = np.append(test_feature, model(alexrank_test_transform(Image.open(eval_root + img).convert('RGB')).unsqueeze(0).to(device))[0].cpu().detach().numpy().reshape(1, -1), axis = 0)


test_score = np.zeros((usernum, len(test_feature)))
for i in tqdm_notebook(range(usernum)):
    temp_dist = np.empty((0, len(test_feature)))
    for idx in item_dict[i]:
        temp_dist = np.append(temp_dist, np.linalg.norm(test_feature - alex_4096_cnn_f[idx], axis = 1).reshape(-1, len(test_feature)), axis = 0)
    test_score[i] = (np.sum(temp_dist, axis = 0) / len(temp_dist))
max_item_num = 100

HR_org_top5_count = []
HR_org_top10_count = []
HR_org_top20_count = []

HR_adv_top5_count = []
HR_adv_top10_count = []
HR_adv_top20_count = []

for i in tqdm_notebook(range(max_item_num)):

    to_test = np.append(bpr_scores, test_score[:, i].reshape(-1, 1), axis = 1)
    rank_res = np.argsort(to_test)

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
np.save('./results/' + 'alexrank_' + item + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
    [np.array(HR_org_top10_count).mean()],
    [np.array(HR_org_top20_count).mean()],
    [np.array(HR_adv_top5_count).mean()],
    [np.array(HR_adv_top10_count).mean()],
    [np.array(HR_adv_top20_count).mean()]]))