import os
import itertools
import torch
import torch.nn as nn
from torch import optim
# from auc_mu import auc_mu
import torch.nn.functional as Func
from datareader import Omniglot, SeqOmniglot, omniglot_load
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
# from loss import GE2ELossWeighted
from tqdm import tqdm
from cnn import resnet18
from initialization import weight_init
import random
import numpy as np
import argparse
import sys

EMBEDDING_DIM = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class F_net (nn.Module):
    def __init__ (self):
        super(F_net, self).__init__()
        self.cnn = resnet18(feat_dim = EMBEDDING_DIM).to(device)

    def forward (self, X):
        x = self.cnn(X)
        x = Func.normalize(x)
        return x

def getDataloader(path):#, class_per_set=5, num_per_set=3, num_per_class=NUM_TRAIN_CLASSES, num_samples=20000):
    dataset = omniglot_load(path)#, class_per_set ,num_per_set, num_per_class, num_samples)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8)
    return dataloader

def get_centroids(embeddings, speaker_split):
    num_spks = len(speaker_split)
    centroids = []
    for i in range(num_spks):
        st = int(np.sum(speaker_split[:i]))
        centroid = torch.mean(embeddings[st:int(st+speaker_split[i])], 0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    return torch.norm(D, p=2, dim=2)

def trainphase(f_net, criterion, optimizer, path='/home/***/data1t_ssd/omniglot/train'):
    f_net.train()
    dataloader = getDataloader(path)
    combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))
    combinations3 = torch.tensor([[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [4, 3], [4, 4], [5, 4], [7, 4]])
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    criterion = nn.MarginRankingLoss(0.1)
    for i, data in enumerate(t):
        test_imgs, ref_imgs = data

        test_imgs, ref_imgs = test_imgs[0].float()/256, ref_imgs[0].float()/256
        test_imgs, ref_imgs = test_imgs.to(device), ref_imgs.to(device)
        imgs = torch.cat([test_imgs, ref_imgs], 0)


        embeddings = f_net(imgs)
        optimizer.zero_grad()

        truth = embeddings[:5]
        preds = embeddings[:-5]
        dists = pairwiseDists(preds, truth)
        losses = torch.zeros(5, device=device)
        for i in range(5):

            dist = dists[i]
            losses[i] = criterion(dist[[e for e in range(5) if e != i]], dist[[i]*4], torch.ones(4, device=device))
        loss = torch.mean(losses)
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(f_net, path='/home/***/data1t_ssd/omniglot/validation'):
    f_net.eval()
    dataLoader = getDataloader(path)
    all_cnt = 0
    hit_cnt = 0
    for i, data in enumerate(dataLoader):
        test_imgs, ref_imgs = data

        test_imgs, ref_imgs = test_imgs[0, :5].float()/256, ref_imgs[0].float()/256
        test_imgs, ref_imgs = test_imgs.to(device), ref_imgs.to(device)
        imgs = torch.cat([test_imgs, ref_imgs], 0)
        embeddings = f_net(imgs)


        centroids = embeddings[-5:]

        truth = centroids[:5]
        preds = embeddings[:-5]
        cosinesim = torch.matmul(preds, truth.transpose(0, 1))
        _, res = torch.topk(cosinesim, 1)
        label = torch.tensor(list(range(5)), device=device)
        all_cnt += 5
        hit_cnt += torch.sum(label == res.squeeze()).item()
    return hit_cnt/all_cnt

def testphase(f_net, path='/home/***/data1t_ssd/omniglot/test_aug_3'):
    f_net.eval()
    dataloader = getDataloader(path)
    all_cnt0 = 0
    top3_hit_cnt0 = 0
    top1_hit_cnt0 = 0
    all_cnt1 = 0
    top3_hit_cnt1 = 0
    top1_hit_cnt1 = 0
    all_cnt2 = 0
    top3_hit_cnt2 = 0
    top1_hit_cnt2 = 0
    # setsize = torch.tensor([1 for i in range(5)] + [2 for i in range(5, 15)] + [3 for i in range(15, 25)], device=device)
    combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))
    combinations3 = torch.tensor([[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [4, 3], [4, 4], [5, 4], [7, 4]])
    for i, data in enumerate(dataloader):
        test_imgs, ref_imgs = data

        test_imgs, ref_imgs = test_imgs[0].float()/256, ref_imgs[0].float()/256
        test_imgs, ref_imgs = test_imgs.to(device), ref_imgs.to(device)
        imgs = torch.cat([test_imgs, ref_imgs], 0)
        embeddings = f_net(imgs)

        centroids = embeddings[-5:]
        comb2_a = centroids[combinations2.transpose(-2, -1)[0]]
        comb2_b = centroids[combinations2.transpose(-2, -1)[1]]
        merged2 = Func.normalize(comb2_a+comb2_b)#g_net(comb2_a, comb2_b)

        comb3_a = merged2[combinations3.transpose(-2, -1)[0]]
        comb3_b = centroids[combinations3.transpose(-2, -1)[1]]
        merged3 = Func.normalize(comb3_a+comb3_b)#g_net(comb3_a, comb3_b)

        truth = torch.cat([centroids, merged2, merged3])
        preds = embeddings[:-5]
        dist = torch.matmul(preds, truth.transpose(0, 1))

        _, res = torch.topk(dist, 1)
        # res = setsize[res].squeeze()
        _, res_top3 = torch.topk(dist, 3)
        # res_top3 = setsize[res_top3]
        # res_top3 = res_top3.data.cpu().numpy()
        for i in range(5):
            if i in res_top3[i]:
                top3_hit_cnt0 += 1
        for i in range(5, 15):
            if i in res_top3[i]:
                top3_hit_cnt1 += 1
        for i in range(15, 25):
            if i in res_top3[i]:
                top3_hit_cnt2 += 1

        label = torch.tensor(list(range(25)), device=device)
        # label = setsize
        all_cnt0 += 5
        top1_hit_cnt0 += torch.sum(label[:5] == res.squeeze()[:5]).item()
        all_cnt1 += 10
        top1_hit_cnt1 += torch.sum(label[5:15] == res.squeeze()[5:15]).item()
        all_cnt2 += 10
        top1_hit_cnt2 += torch.sum(label[15:] == res.squeeze()[15:]).item()
    return (top1_hit_cnt0/all_cnt0, 
        top3_hit_cnt0/all_cnt0, 
        top1_hit_cnt1/all_cnt1, 
        top3_hit_cnt1/all_cnt1, 
        top1_hit_cnt2/all_cnt2, 
        top3_hit_cnt2/all_cnt2,
        (top1_hit_cnt0+top1_hit_cnt1+top1_hit_cnt2)/(all_cnt0+all_cnt1+all_cnt2), 
        (top3_hit_cnt0+top3_hit_cnt1+top3_hit_cnt2)/(all_cnt0+all_cnt1+all_cnt2))

def main(args):
    f_net = F_net().to(device)
    f_net.apply(weight_init)

    if args.load_model:
        checkpoint = torch.load(args.load_model)
        f_net.load_state_dict(checkpoint)
    if args.mode == 'train':
        os.makedirs(args.save_model, exist_ok=True)
        epochs = args.epochs
        criterion = nn.CrossEntropyLoss()#GE2ELossWeighted()
        optimizer = optim.Adam(list(f_net.parameters()))
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

    # f_net.load_state_dict(torch.load("checkpoint_omniglot_3_mean/0.979_f.pt"))


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