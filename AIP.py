
import os
import time
import argparse
import numpy as np
import random

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

import matplotlib.pyplot as plt

from PIL import Image
from io import StringIO, BytesIO


class trainset(Dataset):
    def __init__(self, train_ls):
        self.target = train_ls

    def __getitem__(self, index):
        target = self.target[index]
        return target

    def __len__(self):
        return len(self.target)

orginal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))
    ])

tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

orginal_transform_alex = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),])

partial_transform = transforms.Compose([
    transforms.ToTensor(),])

def INSA_DVBPR(save_root, org_img_num, User_content_embedding, epsilon, Item, device, model, norm):

    image_o = tensor_transform(Image.open(BytesIO(Item[org_img_num][b'imgs']))).to(device).unsqueeze(0)
    delta = torch.rand([1, 3, 224, 224], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([delta], lr=1e-3)

    train_ls = [User_content_embedding[u_idx] for u_idx in range(len(User_content_embedding))]
    train_data  = trainset(train_ls)

    for epoch in range(10):
        for data in DataLoader(train_data, batch_size = 512, shuffle = False, num_workers = 4):
            loss = -1 * torch.sum(model(norm(image_o + delta)) * data.to(device), axis = 1).exp().log().mean()
#             print(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            delta.data = torch.clamp(delta.data, -epsilon/255, epsilon/255)
            delta.data = (torch.clamp(image_o + delta.data, 0 + 1e-6, 1 - 1e-6) - image_o)

    X_new = image_o + delta.data
    x_np = transforms.ToPILImage()((torch.round(X_new[0]*255)/255).detach().cpu())
    x_np.save(save_root + str(org_img_num) +'.png')


def EXPA_DVBPR(org_img_path, target_number, adv_images_root, epsilon, Item, device, model, norm):
    image_t = tensor_transform(Image.open(BytesIO(Item[org_img_path][b'imgs']))).to(device)

    target_feature = model(orginal_transform(Image.open(BytesIO(Item[target_number][b'imgs']))).unsqueeze(0).to(device))
    v = torch.zeros_like(image_t, requires_grad=True, device = device)

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-3
    optimizer = torch.optim.Adam([v], lr=learning_rate)

    for t in tqdm(range(5000)):

        y_pred = model(norm(image_t + v))
        loss = loss_fn(y_pred, target_feature)
        # print(loss.item())

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        v.data = torch.clamp(v.data, -epsilon/255, epsilon/255)
        v.data = (torch.clamp(image_t + v.data, 0 + 1e-6, 1 - 1e-6) - image_t)
    X_new = image_t + v.data
    x_np = transforms.ToPILImage()((torch.round(X_new*255)/255).detach().cpu())
    x_np.save(adv_images_root + str(org_img_path) +'.png')

def EXPA_DVBPR_new(org_img_path, target_number, adv_images_root, epsilon, Item, device, model, norm, alpha=1/255):
    image_t = tensor_transform(Image.open(BytesIO(Item[org_img_path][b'imgs']))).to(device)

    target_feature = model(orginal_transform(Image.open(BytesIO(Item[target_number][b'imgs']))).unsqueeze(0).to(device))
    v = torch.zeros_like(image_t, requires_grad=True, device = device)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    # learning_rate = 1e-2
    # optimizer = torch.optim.Adam([v], lr=learning_rate)

    for t in tqdm(range(1000)):

        y_pred = model(norm(image_t + v))
        loss = loss_fn(y_pred, target_feature)
        # print(loss.item())

        # optimizer.zero_grad()

        loss.backward(retain_graph=True)
        # optimizer.step()
        adv_images = image_t - alpha*torch.sign(v.grad)
        v.data = torch.clamp(adv_images - image_t, -epsilon/255, epsilon/255)
        # v.data = torch.clamp(v.data, -epsilon/255, epsilon/255)
        v.data = (torch.clamp(image_t + v.data, 0 + 1e-6, 1 - 1e-6) - image_t)
    X_new = image_t + v.data
    x_np = transforms.ToPILImage()((torch.round(X_new*255)/255).detach().cpu())
    x_np.save(adv_images_root + str(org_img_path) +'.png')

def INSA_VBPR(save_root, org_img_num, usernum, epsilon, Item, device, VBPRmodel, feature_model, norm):
    delta = torch.rand([1, 3, 224, 224], requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=1e-4)
    train_ls = [list((u_idx, org_img_num)) for u_idx in range(usernum)]
    train_data  = trainset(train_ls)

    for epoch in range(5):
        for data in DataLoader(train_data, batch_size = 256, shuffle = False, num_workers = 4):
            ui, xj = data
            image_o = orginal_transform_alex(Image.open(BytesIO(Item[int(xj[0].numpy())][b'imgs']))).unsqueeze(0).to(device)
            loss = -1 *(VBPRmodel(ui.to(device), xj.to(device), feature_model(norm((image_o + delta))))).exp().sum()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            delta.data = torch.clamp(delta.data, -epsilon/255, epsilon/255)
            delta.data = (torch.clamp(image_o + delta.data, 0 + 1e-6, 1 - 1e-6) - image_o)

    X_new = image_o + delta.data
    x_np = transforms.ToPILImage()((torch.round(X_new[0]*255)/255).detach().cpu())
    x_np.save(save_root + str(org_img_num) +'.png')


def Alex_EXPA(save_root, org_img_num, target_item, usernum, epsilon, Item, device, feature_model, norm):
    delta = torch.rand([1, 3, 224, 224], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([delta], lr=1e-2)

    for epoch in range(5000):
        image_o = orginal_transform_alex(Image.open(BytesIO(Item[org_img_num][b'imgs']))).unsqueeze(0).to(device)
        loss =  torch.norm(feature_model(norm((image_o + delta))) - feature_model(norm(orginal_transform_alex(Image.open(BytesIO(Item[target_item][b'imgs']))).to(device))))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        delta.data = torch.clamp(delta.data, -epsilon/255, epsilon/255)
        delta.data = (torch.clamp(image_o + delta.data, 0 + 1e-6, 1 - 1e-6) - image_o)

    X_new = image_o + delta.data
    x_np = transforms.ToPILImage()((torch.round(X_new[0]*255)/255).detach().cpu())
    x_np.save(save_root + str(org_img_num) +'.png')

def INSA_AlexRank(save_root, org_img_num, alexnet_feature, usernum, epsilon, Item, device, model, norm, user_train):

    item_dict = {}
    for u in tqdm(range(usernum)):
        for j in user_train[u]:
            item_id = j[b'productid']
            if u not in item_dict:
                item_dict[u] = [item_id]
            else:
                item_dict[u].append(item_id)

    delta = torch.rand([1, 3, 224, 224], requires_grad=True, device=device)
    image_o = orginal_transform_alex(Image.open(BytesIO(Item[org_img_num][b'imgs']))).unsqueeze(0).to(device)
    optimizer = torch.optim.Adam([delta], lr=1e-3)

    for epoch in range(1):
        for i in tqdm(range(usernum)):
            loss = torch.norm(model(norm(image_o + delta)) - torch.tensor(alexnet_feature[item_dict[i]]).to(device), dim = 1).mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            delta.data = torch.clamp(delta.data, -epsilon/255, epsilon/255)
            delta.data = (torch.clamp(image_o + delta.data, 0 + 1e-6, 1 - 1e-6) - image_o)

    X_new = image_o + delta.data
    x_np = transforms.ToPILImage()((torch.round(X_new[0]*255)/255).detach().cpu())
    x_np.save(save_root + str(org_img_num) +'.png')
