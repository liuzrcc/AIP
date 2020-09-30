import os
import time
import argparse
import numpy as np
import random

import tqdm
from tqdm import tqdm_notebook
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

import matplotlib.pyplot as plt

from PIL import Image
from io import StringIO, BytesIO


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j



class pthDVBPR(nn.Module):
    def __init__(self, K):

        super(pthDVBPR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(64, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)

        self.pool  = nn.MaxPool2d(2, 2, padding=0)
        self.pool_m  = nn.MaxPool2d(2, 2, padding = 1)

        self.fc1 = nn.Linear(7*7*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, K)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool_m(F.relu(self.conv5(x)))
        x = x.view(-1, 7*7*256)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)
        return x


class User_Em(nn.Module):
    def __init__(self, user_size, dim):

        super().__init__()
        self.W = nn.Parameter(torch.empty([user_size, dim]).uniform_(0, 1/100))

    def forward(self, u):
        return self.W[u]



class pthVBPR(nn.Module):

    def __init__(self, user_num, item_num, factor_num, cnn_feature_dim):

        super(pthVBPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        self.alpha = nn.Embedding(1, 1)

        self.beta_u = nn.Embedding(user_num, 1)
        self.beta_i = nn.Embedding(item_num, 1)

        self.gamma_u = nn.Embedding(user_num, factor_num)
        self.gamma_i = nn.Embedding(item_num, factor_num)

        self.theta_u = nn.Embedding(user_num, factor_num)

        self.E = nn.Embedding(factor_num, cnn_feature_dim)

        self.beta_p = nn.Embedding(1, cnn_feature_dim)


        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        nn.init.normal_(self.gamma_u.weight, std=0.01)
        nn.init.normal_(self.gamma_i.weight, std=0.01)
        nn.init.normal_(self.theta_u.weight, std=0.01)

        nn.init.constant_(self.E.weight, 2 / (cnn_feature_dim * factor_num))
        nn.init.constant_(self.alpha.weight, 0.0)
        nn.init.constant_(self.beta_u.weight, 0.0)
        nn.init.constant_(self.beta_i.weight, 0.0)
        nn.init.constant_(self.beta_p.weight, 0.0)


    def forward(self, user, item_i, cnn_feature_i):
        alpha = self.alpha.weight
        beta_u = self.beta_u(user)
        beta_i = self.beta_i(item_i)

        gamma_u = self.gamma_u(user)
        gamma_i = self.gamma_i(item_i)

        theta_u = self.theta_u(user)
        E = self.E
        beta_p = self.beta_p.weight

        prediction_i = alpha + beta_u.T + beta_i.T + torch.sum(gamma_u*gamma_i, axis = 1) \
        + torch.sum(theta_u * torch.mm(E.weight.double(), cnn_feature_i.T).T, axis = 1) + torch.mm(beta_p.double(), cnn_feature_i.T)
        # print(type(cnn_feature_i.T))
        # print(type(E.weight))

        return prediction_i


class AMR(nn.Module):

    def __init__(self, user_num, item_num, factor_num, cnn_feature_dim):

        super(AMR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        self.alpha = nn.Embedding(1, 1)

        self.beta_u = nn.Embedding(user_num, 1)
        self.beta_i = nn.Embedding(item_num, 1)

        self.gamma_u = nn.Embedding(user_num, factor_num)
        self.gamma_i = nn.Embedding(item_num, factor_num)

        self.theta_u = nn.Embedding(user_num, factor_num)

        self.E = nn.Embedding(factor_num, cnn_feature_dim)

        self.beta_p = nn.Embedding(1, cnn_feature_dim)


        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        nn.init.normal_(self.gamma_u.weight, std=0.01)
        nn.init.normal_(self.gamma_i.weight, std=0.01)
        nn.init.normal_(self.theta_u.weight, std=0.01)

        nn.init.constant_(self.E.weight, 2 / (cnn_feature_dim * factor_num))
        nn.init.constant_(self.alpha.weight, 0.0)
        nn.init.constant_(self.beta_u.weight, 0.0)
        nn.init.constant_(self.beta_i.weight, 0.0)
        nn.init.constant_(self.beta_p.weight, 0.0)


    def forward(self, user, item_i, cnn_feature_i, delta_u=None, delta_i=None, adv = False):
        alpha = self.alpha.weight
        beta_u = self.beta_u(user)
        beta_i = self.beta_i(item_i)

        gamma_u = self.gamma_u(user)
        gamma_i = self.gamma_i(item_i)

        theta_u = self.theta_u(user)
        E = self.E
        beta_p = self.beta_p.weight
        if not adv:
            prediction_i = alpha + beta_u.T + beta_i.T + torch.sum(gamma_u*gamma_i, axis = 1) \
            + torch.sum(theta_u * torch.mm(E.weight, cnn_feature_i.T).T, axis = 1) + torch.mm(beta_p, cnn_feature_i.T)
        else:
            prediction_i = alpha + beta_u.T + beta_i.T + torch.sum((gamma_u+delta_u)*(gamma_i+delta_i), axis = 1) \
            + torch.sum(theta_u * torch.mm(E.weight.double(), cnn_feature_i.T).T, axis = 1) + torch.mm(beta_p.double(), cnn_feature_i.T)
        return prediction_i
