import os, sys
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from datareader import Omniglot, IdealOmniglot, IdealSeqOmniglot, Omniglot1on1, MikeReader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from cnn import resnet18
from initialization import weight_init
import random
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from torch.utils.data.dataset import Dataset
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 32

class omniglot_load(Dataset):
    def __init__(self, path):
        self.path = path
        self.samples = os.listdir(self.path)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        size = int(sample.split('_')[-1])
        root = os.path.join(self.path, sample)
        merge = np.array(imageio.imread(os.path.join(root, 'merge.jpg')))
        pos = np.array(imageio.imread(os.path.join(root, 'pos.jpg')))
        neg = np.array(imageio.imread(os.path.join(root, 'neg.jpg')))
        return merge, pos, neg, size

    def __len__(self):
        return len(self.samples)

class F_net (nn.Module):
    def __init__ (self):
        super(F_net, self).__init__()
        self.cnn = resnet18(feat_dim = EMBEDDING_DIM).to(device)

    def forward (self, X):
        # X = X.unsqueeze(1)
        x = self.cnn(X)
        x = Func.normalize(x)
        return x
        


def getDataloader(path):
    dataset = omniglot_load(path)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=16)
    return dataloader

def trainphase(fnet, criterion, optimizer, path='/home/***/omniglot_mike/train'):
    fnet.train()
    dataloader = getDataloader(path)
    criterion = nn.MarginRankingLoss(0.1)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        
        optimizer.zero_grad()
        test_imgs, pos_imgs, neg_imgs, _ = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256

        embeddings = fnet(imgs)

        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]
        labels = torch.tensor([-1 for e in range(valid_batch_size)], device=device).float()
        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        loss = criterion(dists[:valid_batch_size], dists[valid_batch_size:], labels)
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(fnet, path):
    torch.manual_seed(0)
    fnet.eval()
    dataLoader = getDataloader(path)
    hit_cnt = 0
    all_cnt = 0
    ytrue= []
    ypred = []
    
    for i, data in enumerate(dataLoader):
        test_imgs, pos_imgs, neg_imgs, _ = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256

        embeddings = fnet(imgs)

        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]

        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        ypred.append(torch.sigmoid(1-dists.squeeze()))
        pred = dists.squeeze() < 0.5
        ytrue += labels
        hit_cnt += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt += valid_batch_size*2
    ypred = torch.cat(ypred, 0)
    roc=roc_auc_score(np.array(ytrue), ypred.data.cpu().numpy())
    return hit_cnt/all_cnt, roc

def testphase(fnet, path):
    fnet.eval()
    dataloader = getDataloader(path)

    hit_cnt = [0]*5
    all_cnt = [0]*5
    ytrue = [[], [], [], [], []]
    ypred = [[], [], [], [], []]
    hit_cnt_all = 0
    all_cnt_all = 0
    ytrue_all = []
    ypred_all = []
    rocs = []
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        test_imgs, pos_imgs, neg_imgs, num_speakers = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256
        embeddings = fnet(imgs)

        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]

        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        pred = 1-dists.squeeze()

        # pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        # import pdb;pdb.set_trace()
        pred_b = pred.squeeze() > 0.6
        # import pdb;pdb.set_trace()
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
        pred = pred.squeeze() > 0.6
        ytrue_all += labels
        hit_cnt_all += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt_all += valid_batch_size*2
    # import pdb;pdb.set_trace()
    for j in range(5):
        rocs.append(roc_auc_score(np.array(ytrue[j]), np.array(ypred[j])))
        
    ypred_all = torch.cat(ypred_all, 0)
    roc=roc_auc_score(np.array(ytrue_all), ypred_all.data.cpu().numpy())
    return hit_cnt, all_cnt, rocs, hit_cnt_all/all_cnt_all, roc


def main(args):
    f_net = F_net().to(device)
    f_net.apply(weight_init)

    if args.load_model:
        checkpoint = torch.load(args.load_model)
        f_net.load_state_dict(checkpoint)
    if args.mode == 'train':
        os.makedirs(args.save_model, exist_ok=True)
        epochs = args.epochs
        criterion = nn.BCELoss()#GE2ELossWeighted()
        optimizer = optim.Adam(list(f_net.parameters()), lr=3e-4)
        for epoch in range(epochs):
            loss = trainphase(f_net, criterion, optimizer, args.train_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            with torch.no_grad():
                accu = validationphase(f_net, args.val_dir)
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save(f_net.state_dict(), os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        with torch.no_grad():
            accu = testphase(f_net, args.test_dir)
        print('test accu is {}'.format(accu))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['train', 'test'],
        help='select to train a new model or test', default='test')
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