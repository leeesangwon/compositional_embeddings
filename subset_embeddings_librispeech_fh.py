import os, sys
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from datareader import LibriReadeload
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from h_net import *
from tqdm import tqdm
from cnn import resnet18
from initialization import weight_init
import random
import itertools
import numpy as np
import torchvision.transforms.functional as VF
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 32

class Lstm(nn.Module):
    def __init__(self, model_dim=30):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(model_dim, 256, 2, batch_first = True)
        # self.layer0 = Encoder(max_seq_len = length, num_layers = 1, model_dim = model_dim, position = True, num_heads=num_heads)
        self.outlayer = nn.Linear(256, EMBEDDING_DIM)
        
        
    def forward(self, inputs, mask=None):
        out, _ = self.lstm(inputs)
        out = out[:,-1, :]
        out = self.outlayer(Func.leaky_relu(out))
        out = Func.normalize(out)
        return out
        


def getDataloader(path, length=2, batch_size=32, dataset_length=10000):
    dataset = LibriReadeload(path) #LibriReader(path, length, dataset_length)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    return dataloader

def trainphase(fnet, gnet, criterion, optimizer, batch_size, path='/home/***/data1t_ssd/librispeech_mikenet/train'):
    fnet.train()
    gnet.train()
    dataloader = getDataloader(path, batch_size=batch_size)#(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length = 2, batch_size = batch_size, dataset_length = 100000)

    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))

    for i, data in enumerate(t):
        
        optimizer.zero_grad()
        test_imgs, pos_imgs, neg_imgs, _ = data
        # import pdb;pdb.set_trace()
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        # imgs = imgs.permute(0, 2, 1)
        # imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device).float()
        embeddings = fnet(imgs)
        # import pdb;pdb.set_trace()
        pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        loss = criterion(pred.squeeze(), torch.tensor(labels, device=device).float())
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(fnet, gnet, batch_size, path='/home/***/data1t_ssd/librispeech_mikenet/test'):
    fnet.eval()
    gnet.eval()
    dataloader = getDataloader(path, batch_size=batch_size)#(path='/home/***/data1t_ssd/LibriSpeech/devSpker', length = 2, batch_size = batch_size, dataset_length = 10000)

    hit_cnt = 0
    all_cnt = 0
    ytrue = []
    ypred = []
    for i, data in enumerate(dataloader):
        test_imgs, pos_imgs, neg_imgs, num_speakers = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        # imgs = imgs.permute(0, 2, 1)
        imgs = imgs.to(device).float()
        embeddings = fnet(imgs)
        pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        # import pdb;pdb.set_trace()
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        ypred.append(pred.squeeze())
        pred = pred.squeeze() > 0.5
        ytrue += labels
        hit_cnt += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt += valid_batch_size*2
    ypred = torch.cat(ypred, 0)
    roc=roc_auc_score(np.array(ytrue), ypred.data.cpu().numpy())
    return hit_cnt/all_cnt, roc

def testphase(fnet, gnet, batch_size, path='/home/***/data1t_ssd/librispeech_mikenet/test'):
    fnet.eval()
    gnet.eval()
    dataloader = getDataloader(path, batch_size=batch_size)

    hit_cnt = [0]*5
    all_cnt = [0]*5
    ytrue = [[], [], [], [], []]
    ypred = [[], [], [], [], []]
    hit_cnt_all = 0
    all_cnt_all = 0
    ytrue_all = []
    ypred_all = []
    rocs = []
    for i, data in enumerate(dataloader):
        test_imgs, pos_imgs, neg_imgs, num_speakers = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        # imgs = imgs.permute(0, 2, 1)
        imgs = imgs.to(device).float()
        embeddings = fnet(imgs)
        pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        # import pdb;pdb.set_trace()
        pred_b = pred.squeeze() > 0.5
        for j in range(valid_batch_size):
            if pred_b[j]:
                hit_cnt[num_speakers[j]-1] += 1
            all_cnt[num_speakers[j]-1] += 1
            ytrue[num_speakers[j]-1].append(1)
            ypred[num_speakers[j]-1].append(pred[j].cpu().item())
        
        for j in range(valid_batch_size):
            if pred_b[j+valid_batch_size] == 0:
                hit_cnt[num_speakers[j]-1] += 1
            all_cnt[num_speakers[j]-1] += 1
            ytrue[num_speakers[j]-1].append(0)
            ypred[num_speakers[j]-1].append(pred[j+valid_batch_size].cpu().item())

        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        ypred_all.append(pred.squeeze())
        pred = pred.squeeze() > 0.5
        ytrue_all += labels
        hit_cnt_all += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt_all += valid_batch_size*2
    # import pdb;pdb.set_trace()
    for j in range(5):
        rocs.append(roc_auc_score(np.array(ytrue[j]), np.array(ypred[j])))
        
    # ypred_all = torch.cat(ypred_all, 0)
    # roc=roc_auc_score(np.array(ytrue_all), ypred_all.data.cpu().numpy())
    return hit_cnt, all_cnt, rocs

def main(args):
    f_net = Lstm(32).to(device)
    if args.g == 'dnn':
        mike_net = Mike_net_dnn(EMBEDDING_DIM).to(device)
    elif args.g == 'linear_fc':
        mike_net = Mike_net_linear_fc(EMBEDDING_DIM).to(device)
    elif args.g == 'linear':
        mike_net = Mike_net_linear(EMBEDDING_DIM).to(device)
    f_net.apply(weight_init)
    mike_net.apply(weight_init)

    if args.load_model:
        checkpoint = torch.load(args.load_model)
        f_net.load_state_dict(checkpoint['f_net'])
        mike_net.load_state_dict(checkpoint['g_net'])
    if args.mode == 'train':
        os.makedirs(args.save_model, exist_ok=True)
        epochs = args.epochs
        criterion = nn.BCELoss()#GE2ELossWeighted()
        optimizer = optim.Adam(list(f_net.parameters())+list(mike_net.parameters()), lr=3e-4)
        for epoch in range(epochs):
            loss = trainphase(f_net, mike_net, criterion, optimizer, 128, args.train_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            with torch.no_grad():
                accu = validationphase(f_net, mike_net, 128, args.val_dir)
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save({'f_net':f_net.state_dict(), 'g_net':mike_net.state_dict()}, os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        with torch.no_grad():
            accu = testphase(f_net, mike_net, 128, args.test_dir)
        print('test accu is {}'.format(accu))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['train', 'test'],
        help='select to train a new model or test', default='test')
    parser.add_argument('g', type=str, choices=['dnn', 'linear_fc', 'linear', 'mean'],
        help='select which g to use')
    parser.add_argument('--train_dir', type=str,
        help='Path to the data directory containing training data')
    parser.add_argument('--val_dir', type=str,
        help='Path to the data directory containing validation data')
    parser.add_argument('--test_dir', type=str,
        help='Path to the data directory containing test data')
    parser.add_argument('--save_model', type=str, 
        help='Directory Path of saving model checkpoint')
    parser.add_argument('--load_model', type=str, 
        help='Path of loading model checkpoint')
    parser.add_argument('--epochs', type=int,
        help='number of epochs in training')
    
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))