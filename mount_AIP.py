import os
import time
import argparse
import numpy as np
import random
from PIL import Image
from io import StringIO, BytesIO

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F

from model import pthDVBPR, pthVBPR, Normalize
from AIP import INSA_DVBPR, EXPA_DVBPR, INSA_VBPR, Alex_EXPA, INSA_AlexRank, EXPA_DVBPR_new



parser = argparse.ArgumentParser(description = "AIP attacks")
parser.add_argument("-data_set", "--data_for_experiment", help="Training data to use", default="amazon")
parser.add_argument("-gpu_id", "--gpu", type=int, help="Using GPU or not, cpu please use -1", default='0')
parser.add_argument("-model_to_attack", "--model_to_attack", help="Length of latent factors", default="DVBPR")
parser.add_argument("-attack_type", "--attack_type", help="Length of latent factors", default="INSA")
parser.add_argument("-L_inf_norm", "--epsilon", type=int, help="Length of latent factors", default="32")
args = parser.parse_args()


data_for_experiment = args.data_for_experiment
if args.gpu == 0:
    device = 'cuda:0'
elif args.gpu == -1:
    device = 'cpu'

attack_type = args.attack_type
model_to_attack = args.model_to_attack
epsilon = args.epsilon


print('######## Loading data ########')
data_root = './data/'
if data_for_experiment == 'amazon':
    dataset_name = 'AmazonMenWithImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_k = np.load(data_root + 'amazon_one_k_cold.npy')

    alex_4096_cnn_f = np.load(data_root + 'amazon_alexnet_features.npy')
elif data_for_experiment == 'tradesy':
    dataset_name = 'TradesyImgPartitioned.npy'

    dataset = np.load(data_root + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

    alex_4096_cnn_f = np.load(data_root + 'tradesy_alexnet_features.npy')
    cold_k = np.load(data_root + 'tradesy_one_k_cold.npy')

print('######## Data Loaded ########')


print('######## Loading recsys models ########')

if model_to_attack == 'DVBPR':
    model_name = data_for_experiment + '_k100_' + model_to_attack + '.tar'
    model = pthDVBPR(100)

    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    dvbpr_U = checkpoint['U']
    model.to(device).eval()

    norm = Normalize(mean=[0.6949, 0.6748, 0.6676], std=[0.3102, 0.3220, 0.3252])
    norm.to(device)

elif model_to_attack == 'VBPR':
    model_name = data_for_experiment + '_k100_' + model_to_attack + '.pt'


    model = pthVBPR(usernum, itemnum, 100, 4096).to(device)
    model = torch.load('./models/' + model_name, map_location=device)
    model.eval().to(device)
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm.to(device)

    feature_model = models.alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(feature_model.classifier.children())[:-1])
    feature_model.classifier = new_classifier
    feature_model.eval().to(device)

elif model_to_attack == 'AMR':
    model_name = data_for_experiment + '_k100_' + model_to_attack + '.pt'


    model = pthVBPR(usernum, itemnum, 100, 4096).to(device)
    model = torch.load('./models/' + model_name, map_location=device)
    model.eval().to(device)
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm.to(device)

    feature_model = models.alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(feature_model.classifier.children())[:-1])
    feature_model.classifier = new_classifier
    feature_model.eval().to(device)

elif model_to_attack == 'AlexRank':
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm.to(device)
    feature_model = models.alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(feature_model.classifier.children())[:-1])
    feature_model.classifier = new_classifier
    feature_model.eval().to(device)

print('######## Models loaded ########')


print('######## Mounting ' + attack_type + ' attatck on BPR + ' + model_to_attack +  '########')

save_root = './adv_output/' +  model_to_attack + '/' + data_for_experiment + '_'  + attack_type + '/'
if not os.path.isdir(save_root):
        os.mkdir(save_root)

if model_to_attack == 'DVBPR':
    if attack_type == 'INSA':
        for item in tqdm(cold_k):
            INSA_DVBPR(save_root, item, dvbpr_U, epsilon, Item, device, model, norm)
    elif attack_type == 'EXPA':
        for item in tqdm(cold_k):
            EXPA_DVBPR(item, 901, save_root, epsilon, Item, device, model, norm)


if model_to_attack == 'VBPR':
    if attack_type == 'INSA':
        for cold_i in tqdm(cold_k):
            INSA_VBPR(save_root, cold_i, usernum, epsilon, Item, device, model, feature_model, norm)
    elif attack_type == 'EXPA':
        for cold_i in tqdm(cold_k):
            if data_for_experiment == 'amazon':
                Alex_EXPA(save_root, cold_i, 901, usernum, epsilon, Item, device, feature_model, norm)
                

if model_to_attack == 'AMR':
    if attack_type == 'INSA':
        for cold_i in tqdm(cold_k):
            INSA_VBPR(save_root, cold_i, usernum, epsilon, Item, device, model, feature_model, norm)
    elif attack_type == 'EXPA':
        for cold_i in tqdm(cold_k):
            if data_for_experiment == 'amazon':
                Alex_EXPA(save_root, cold_i, 901, usernum, epsilon, Item, device, feature_model, norm)


if model_to_attack == 'AlexRank':
    if attack_type == 'INSA':
        for cold_i in tqdm(cold_k):
            INSA_AlexRank(save_root, cold_i, alex_4096_cnn_f, usernum, epsilon, Item, device, feature_model, norm, user_train)
    elif attack_type == 'EXPA':
        for cold_i in tqdm(cold_k):
            Alex_EXPA(save_root, cold_i, 901, usernum, epsilon, Item, device, feature_model, norm)



print('######## Attack finished! ########')
