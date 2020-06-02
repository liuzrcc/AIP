import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
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


model = pthDVBPR(100)
checkpoint = torch.load('./models/' + data_for_experiment + '_k100_DVBPR.tar')
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device).eval()


for param in model.parameters():
    param.requires_grad = False
    
DVBPR_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))
    ])

input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

dvbpr_U = checkpoint['U']
perturb_items = np.zeros((100, 100))
bpr_scores = np.load('./intermediate_score_index/DVBPR_' + data_for_experiment + '_k100.npz')['temp_res']


eval_root = './ablation_table_2/DVBPR/' + item + '/'
for idx, img in tqdm(enumerate(os.listdir(eval_root))):
    if img.split('.')[-1] == 'png':
        perturb_items[idx] = model(DVBPR_test_transform(Image.open(eval_root + str(img)).convert('RGB')).to(device).unsqueeze(0)).cpu().data.numpy()[0]
        DVBPR_UI_adv_tar = np.dot(dvbpr_U, perturb_items.T)
        HR_org_top5_count = []
        HR_org_top10_count = []
        HR_org_top20_count = []

        HR_adv_top5_count = [] 
        HR_adv_top10_count = []
        HR_adv_top20_count = []

        for i in tqdm(range(len(os.listdir(eval_root)))):
            to_test = np.append(bpr_scores, DVBPR_UI_adv_tar[:, i].reshape(-1, 1), axis = 1)
            rank_res = np.argsort(-1 * to_test)

            HR_org_top5_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :5]])])
            HR_org_top10_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :10]])])
            HR_org_top20_count.extend([np.array([1000 in rlist for rlist in rank_res[:, :20]])])

            HR_adv_top5_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :5]])])
            HR_adv_top10_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :10]])])
            HR_adv_top20_count.extend([np.array([1001 in rlist for rlist in rank_res[:, :20]])])

        print('------------------------------')
        print(np.array(HR_org_top5_count).mean())
        print(np.array(HR_org_top10_count).mean())
        print(np.array(HR_org_top20_count).mean())
        print(np.array(HR_adv_top5_count).mean())
        print(np.array(HR_adv_top10_count).mean())
        print(np.array(HR_adv_top20_count).mean())
        print('------------------------------')
        np.save('./results/' + 'DVBPR_' + item + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
        [np.array(HR_org_top10_count).mean()],
        [np.array(HR_org_top20_count).mean()],
        [np.array(HR_adv_top5_count).mean()],
        [np.array(HR_adv_top10_count).mean()],
        [np.array(HR_adv_top20_count).mean()]]))