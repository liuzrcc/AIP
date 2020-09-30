import numpy as np
import random
from PIL import Image
from model import pthDVBPR, pthVBPR
import argparse
from tqdm import tqdm
from io import StringIO, BytesIO

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torchvision import transforms, utils


parser = argparse.ArgumentParser(description = "Score and index")
parser.add_argument("-task", "--task", help="Which model to socre (or index). You need to run BPR-DVBPR before VBPR and AlexRank", default="BPR-DVBPR")
parser.add_argument("-data_set", "--data_train", help="Data set to use", default="amazon")
parser.add_argument("-gpu_id", "--gpu", type=int, help="Using GPU or not, cpu please use -1", default='0')
parser.add_argument("-model_path", "--model_path", help="Path to load trained model.", default='./models/')
parser.add_argument("-score_path", "--score_path", help="Path to save calculated scores and index.", default='./bpr_score_index/')
args = parser.parse_args()


if args.gpu == 0:
    device = 'cuda:0'
elif args.gpu == -1:
    device = 'cpu'


data_train = args.data_train

if data_train == 'amazon':

    dataset_name = 'AmazonMenWithImgPartitioned.npy'

    dataset = np.load('./data/'+ dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('./data/amazon_one_k_cold.npy')
    alex_4096_cnn_f = np.load('./data/amazon_alexnet_features.npy')
elif data_train == 'tradesy':

    dataset_name = 'TradesyImgPartitioned.npy'
    dataset = np.load('./data/' + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('./data/tradesy_one_k_cold.npy')
    alex_4096_cnn_f = np.load('./data/tradesy_alexnet_features.npy')


if args.task == 'BPR-DVBPR':

    print('loaded data from' + data_train)

    model_stage_1 = torch.load(args.model_path + '/BPR_amazon.pt')
    model_stage_1.to(device)

    params = []
    for item in model_stage_1.parameters():
        params.append(item)

    BPR_UI_m = torch.mm(params[0].data.cpu(), params[1].data.cpu().T).numpy()

    print('BPR scores is calculated.')

    def sample_for_test(user, user_index):
        u = user_index
        ls_i = [user[u][0][b'productid']]
        M=set()
        for item in user_train[u]:
            M.add(item[b'productid'])

        for i in ls_i:
            while True:
    #             random.seed(1234)
                j=random.randrange(itemnum)
                if (not j in M): break
        return (u, i, j)


    class testset(Dataset):
        def __init__(self, test_ls):
            self.target = test_ls

        def __getitem__(self, index):
            target = self.target[index]
            return target[0], target[1], target[2]

        def __len__(self):
            return len(self.target)


    test_ls = [list(sample_for_test(user_test, u)) for u in range(usernum)]

    test_data = testset(test_ls)
    test_loader = DataLoader(test_data, batch_size = 256,
                           shuffle = False, num_workers = 2)


    st1_1000 = np.empty(shape=[0, 1000])
    for user, item_i, item_j in tqdm(test_loader):

        user = user
        gt_item = item_i
        temp_res = BPR_UI_m[user.cpu().numpy()]
        st1_1000 =  np.append(st1_1000, np.argsort(-1 * temp_res)[:, :1000], axis = 0)

#     np.save('./bpr_score_index/st1000_' + data_train +'.npy', st1000)
#     print('candidate generated')
####################################################################################
    # calculation of DVBPR scores and BPR indexes

    model = pthDVBPR(100)
    input_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))])

    checkpoint = torch.load(args.model_path + data_train + '_k100_DVBPR.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    dvbpr_U = checkpoint['U']

    def default_loader(path):
        img_pil =  Image.open(BytesIO(path)).convert('RGB')
        img_tensor = input_transform(img_pil)
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

    file_item_i = [Item[i][b'imgs'] for i in range(itemnum)]
    item_data  = testset_img(file_item_i)

    dvbpr_I = np.array([])
    for data in tqdm(DataLoader(item_data, batch_size = 128, num_workers = 4)):
        if len(dvbpr_I) == 0:
            dvbpr_I = model(data.to(device)).cpu().data.numpy()
        else:
            dvbpr_I = np.append(dvbpr_I, model(data.to(device)).cpu().data.numpy(), axis = 0)


    DVBPR_UI_m = np.dot(dvbpr_U, dvbpr_I.T)

    test_add_final = np.empty([0, 1001])
    temp_res_final = np.empty([0, 1001])

    for user, item_i, item_j in tqdm(test_loader):
        user = user
        gt_item = item_i
        st1_1000_r = st1_1000[user.cpu().numpy()]
        test_add = np.append(np.array(st1_1000_r), np.array(gt_item.cpu().numpy()).reshape(len(gt_item), 1), axis = 1)

        temp_res = DVBPR_UI_m[[[i] for i in user.cpu().numpy()], [np.array(test_add).astype(np.int)]][0]
        test_add_final = np.append(test_add_final, test_add, axis = 0)
        temp_res_final = np.append(temp_res_final, temp_res, axis = 0)

    np.save(args.score_path + 'DVBPR_'+ data_train +'_k100.npy', temp_res_final)
    np.save(args.score_path + 'bpr_'+ data_train +'_index.npy', test_add_final)


####################################################################################
if args.task == 'VBPR':
    # calculation of VBPR scores
    bpr_index = np.load(args.score_path + 'bpr_' + data_train + '_index.npy')
    model = pthVBPR(usernum, itemnum, 100, 4096).to(device)
    model = torch.load(args.model_path + data_train + '_k100_VBPR.pt')
    model.eval().to(device)

    VBPR_UI_m_BPR = np.zeros((usernum, 1001))
    for i in tqdm(range(usernum)):
        x1 = torch.LongTensor(np.repeat(i, 1001).astype(np.int32))
        x2 = torch.LongTensor(np.array(bpr_index[i]).astype(np.int32))
        x3 = torch.tensor(alex_4096_cnn_f[np.array(bpr_index[i]).astype(np.int32)]).float()
        res = model(x1.to(device), x2.to(device), x3.to(device))
        VBPR_UI_m_BPR[i] = res.detach().cpu().numpy()[0]



    np.save(args.score_path + 'VBPR_' + data_train + '_k100.npy', VBPR_UI_m_BPR)

####################################################################################
if args.task == 'AMR':
    # calculation of VBPR scores
    bpr_index = np.load(args.score_path + 'bpr_' + data_train + '_index.npy')
    model = AMR(usernum, itemnum, 100, 4096).to(device)
    model = torch.load(args.model_path + data_train + '_k100_AMR.pt')
    model.eval().to(device)

    VBPR_UI_m_BPR = np.zeros((usernum, 1001))
    for i in tqdm(range(usernum)):
        x1 = torch.LongTensor(np.repeat(i, 1001).astype(np.int32))
        x2 = torch.LongTensor(np.array(bpr_index[i]).astype(np.int32))
        x3 = torch.tensor(alex_4096_cnn_f[np.array(bpr_index[i]).astype(np.int32)]).float()
        res = model(x1.to(device), x2.to(device), x3.to(device))
        VBPR_UI_m_BPR[i] = res.detach().cpu().numpy()[0]



    np.save(args.score_path + 'AMR_' + data_train + '_k100.npy', VBPR_UI_m_BPR)

####################################################################################
if args.task == 'AlexRank':
    # calculation of AlexRank scores
    bpr_index = np.load(args.score_path + 'bpr_' + data_train + '_index.npy')
    item_dict = {}
    for u in tqdm(range(usernum)):
        for j in user_train[u]:
            item_id = j[b'productid']
            if u not in item_dict:
                item_dict[u] = [item_id]
            else:
                item_dict[u].append(item_id)

    Visrank_score = np.zeros((usernum, 1001))
    for i in tqdm(range(usernum)):
        temp_dist = np.empty((0, 1001))
        for idx in item_dict[i]:
            temp_dist = np.append(temp_dist, np.linalg.norm(alex_4096_cnn_f[bpr_index[i].astype(np.int32)] - alex_4096_cnn_f[idx], axis = 1).reshape(-1, 1001), axis = 0)
        Visrank_score[i] = (np.sum(temp_dist, axis = 0) / len(temp_dist))


    np.save(args.score_path + 'alexrank_'+ data_train + '_k100.npy', Visrank_score)
