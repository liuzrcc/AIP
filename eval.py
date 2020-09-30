import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO, BytesIO
import threading
from tqdm import tqdm

from model import pthDVBPR, pthVBPR, Normalize


parser = argparse.ArgumentParser(description = "Evaluation")
parser.add_argument("-model_to_eval", "--model_to_eval", help="Which model to eval. ", default="DVBPR")
parser.add_argument("-data_set", "--data_for_experiment", help="Data set to use", default="amazon")
parser.add_argument("-gpu_id", "--gpu", type=int, help="Using GPU or not, cpu please use -1", default='0')
parser.add_argument("-adv_item_path", "--adv_item_path", help="Path to generated adversarial item images.", default='amazon_INSA')
parser.add_argument("-score_path", "--score_path", help="Path to pre-calculated scores.", default='./bpr_score_index/')
args = parser.parse_args()



if args.gpu == 0:
    device = 'cuda:0'
elif args.gpu == -1:
    device = 'cpu'




data_for_experiment = args.data_for_experiment

if args.model_to_eval == 'DVBPR':

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
    perturb_items = np.zeros((1000, 100))
    bpr_scores = np.load(args.score_path + 'DVBPR_' + data_for_experiment + '_k100.npy')


    eval_root = './adv_output/DVBPR/' + args.adv_item_path + '/'
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
    np.save('./results/' + 'DVBPR_' + args.adv_item_path + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
    [np.array(HR_org_top10_count).mean()],
    [np.array(HR_org_top20_count).mean()],
    [np.array(HR_adv_top5_count).mean()],
    [np.array(HR_adv_top10_count).mean()],
    [np.array(HR_adv_top20_count).mean()]]))

################################################################################################
if args.model_to_eval == 'VBPR':

    data_root = './data/'
    if data_for_experiment == 'amazon':
        dataset_name = 'AmazonMenWithImgPartitioned.npy'

        dataset = np.load(data_root + dataset_name, encoding='bytes')
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

        alex_4096_cnn_f = np.load('./data/amazon_alexnet_features.npy')
    elif data_for_experiment == 'tradesy':
        dataset_name = 'TradesyImgPartitioned.npy'

        dataset = np.load(data_root + dataset_name, encoding='bytes')
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

        alex_4096_cnn_f = np.load('./data/tradesy_alexnet_features.npy')


    model = torch.load('./models/'+ data_for_experiment + '_k100_VBPR.pt', map_location = device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False


    VBPR_UI_m_BPR = np.load(args.score_path + '/VBPR_'+ data_for_experiment + '_k100.npy')

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
    bpr_score = np.load(args.score_path + '/VBPR_' + data_for_experiment + '_k100.npy')



    eval_root = './adv_output/VBPR/' + args.adv_item_path + '/'
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
    for data in tqdm(DataLoader(item_data, batch_size = 128, num_workers = 4)):
        if len(I) == 0:
            I = new_model(data.to(device)).cpu().data.numpy()
        else:
            I = np.append(I, new_model(data.to(device)).cpu().data.numpy(), axis = 0)

    j_list = np.array([int(img.split('/')[-1].split('.')[0]) for img in file_item_i])

    scores_temp = np.zeros((usernum, len(j_list)))

    scores_temp = np.zeros((usernum, len(I)))
    for i in tqdm(range(usernum)):
        x1 = torch.LongTensor(np.repeat(i, len(I)).astype(np.int32))
        x2 = torch.LongTensor(np.array(j_list).astype(np.int32))
        x3 = torch.tensor(I)
        res = model(x1.to(device), x2.to(device), x3.to(device))
        scores_temp[i] = res.detach().cpu().numpy()[0]
    max_item_num = 1000

    HR_org_top5_count = []
    HR_org_top10_count = []
    HR_org_top20_count = []

    HR_adv_top5_count = []
    HR_adv_top10_count = []
    HR_adv_top20_count = []

    for i in tqdm(range(max_item_num)):

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
    np.save('./results/' + 'VBPR_' + args.adv_item_path + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
        [np.array(HR_org_top10_count).mean()],
        [np.array(HR_org_top20_count).mean()],
        [np.array(HR_adv_top5_count).mean()],
        [np.array(HR_adv_top10_count).mean()],
        [np.array(HR_adv_top20_count).mean()]]))

################################################################################################
if args.model_to_eval == 'AMR':

    data_root = './data/'
    if data_for_experiment == 'amazon':
        dataset_name = 'AmazonMenWithImgPartitioned.npy'

        dataset = np.load(data_root + dataset_name, encoding='bytes')
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

        alex_4096_cnn_f = np.load('./data/amazon_alexnet_features.npy')
    elif data_for_experiment == 'tradesy':
        dataset_name = 'TradesyImgPartitioned.npy'

        dataset = np.load(data_root + dataset_name, encoding='bytes')
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

        alex_4096_cnn_f = np.load('./data/tradesy_alexnet_features.npy')


    model = torch.load('./models/'+ data_for_experiment + '_k100_AMR.pt', map_location = device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False


    VBPR_UI_m_BPR = np.load(args.score_path + '/AMR_'+ data_for_experiment + '_k100.npy')

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
    bpr_score = np.load(args.score_path + '/AMR_' + data_for_experiment + '_k100.npy')



    eval_root = './adv_output/AMR/' + args.adv_item_path + '/'
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
    for data in tqdm(DataLoader(item_data, batch_size = 128, num_workers = 4)):
        if len(I) == 0:
            I = new_model(data.to(device)).cpu().data.numpy()
        else:
            I = np.append(I, new_model(data.to(device)).cpu().data.numpy(), axis = 0)

    j_list = np.array([int(img.split('/')[-1].split('.')[0]) for img in file_item_i])

    scores_temp = np.zeros((usernum, len(j_list)))

    scores_temp = np.zeros((usernum, len(I)))
    for i in tqdm(range(usernum)):
        x1 = torch.LongTensor(np.repeat(i, len(I)).astype(np.int32))
        x2 = torch.LongTensor(np.array(j_list).astype(np.int32))
        x3 = torch.tensor(I)
        res = model(x1.to(device), x2.to(device), x3.to(device))
        scores_temp[i] = res.detach().cpu().numpy()[0]

    max_item_num = 1000

    HR_org_top5_count = []
    HR_org_top10_count = []
    HR_org_top20_count = []

    HR_adv_top5_count = []
    HR_adv_top10_count = []
    HR_adv_top20_count = []

    for i in tqdm(range(max_item_num)):

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
    np.save('./results/' + 'AMR_' + args.adv_item_path + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
        [np.array(HR_org_top10_count).mean()],
        [np.array(HR_org_top20_count).mean()],
        [np.array(HR_adv_top5_count).mean()],
        [np.array(HR_adv_top10_count).mean()],
        [np.array(HR_adv_top20_count).mean()]]))
################################################################################################
if args.model_to_eval == 'AlexRank':

    if data_for_experiment == 'amazon':
        dataset_name = 'AmazonMenWithImgPartitioned.npy'

        dataset = np.load('./data/' + dataset_name, encoding='bytes')
        [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset

        alex_4096_cnn_f = np.load('./data/amazon_alexnet_features.npy')
    elif data_for_experiment == 'tradesy':
        dataset_name = 'TradesyImgPartitioned.npy'

        dataset = np.load('./data/' + dataset_name, encoding='bytes')
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
    for u in tqdm(range(usernum)):
        for j in user_train[u]:
            item_id = j[b'productid']
            if u not in item_dict:
                item_dict[u] = [item_id]
            else:
                item_dict[u].append(item_id)

    bpr_scores = np.load(args.score_path+ 'alexrank_' + data_for_experiment + '_k100.npy')


    eval_root = './adv_output/AlexRank/' + args.adv_item_path + '/'

    test_feature = np.empty((0, 4096))
    for img in tqdm(os.listdir(eval_root)):
        if img.split('.')[-1] == 'png':
            test_feature = np.append(test_feature, model(alexrank_test_transform(Image.open(eval_root + img).convert('RGB')).unsqueeze(0).to(device))[0].cpu().detach().numpy().reshape(1, -1), axis = 0)


    test_score = np.zeros((usernum, len(test_feature)))
    for i in tqdm(range(usernum)):
        temp_dist = np.empty((0, len(test_feature)))
        for idx in item_dict[i]:
            temp_dist = np.append(temp_dist, np.linalg.norm(test_feature - alex_4096_cnn_f[idx], axis = 1).reshape(-1, len(test_feature)), axis = 0)
        test_score[i] = (np.sum(temp_dist, axis = 0) / len(temp_dist))
    max_item_num = len(test_score)

    HR_org_top5_count = []
    HR_org_top10_count = []
    HR_org_top20_count = []

    HR_adv_top5_count = []
    HR_adv_top10_count = []
    HR_adv_top20_count = []

    for i in tqdm(range(1000)):

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
    np.save('./results/' + 'alexrank_' + args.adv_item_path + '.npy', np.array([[np.array(HR_org_top5_count).mean()],
        [np.array(HR_org_top10_count).mean()],
        [np.array(HR_org_top20_count).mean()],
        [np.array(HR_adv_top5_count).mean()],
        [np.array(HR_adv_top10_count).mean()],
        [np.array(HR_adv_top20_count).mean()]]))
