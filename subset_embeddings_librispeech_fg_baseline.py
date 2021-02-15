import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from datareader import Omniglot, SeqOmniglot, LibriReader_fg
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm
from cnn import resnet18
from initialization import weight_init
import random
import itertools
import numpy as np
import os, sys
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 32

class FeatureReader(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=10000):
        self.path = path
        self.files = os.listdir(path)

    def __getitem__(self, index):
        res_mfcc = torch.load(os.path.join(self.path, self.files[index]))

        return res_mfcc

    def __len__(self):
        return len(self.files)

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

def getDataloader(path, length, batch_size, dataset_length):
    dataset = FeatureReader(path)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=12)
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

def getCombination(num_per_classes=5, num_per_set=2):
    combs = []
    for i in range(2, num_per_set+1):
        combs.extend(list(itertools.combinations(list(range(num_per_classes)),i)))
    combs_array = np.ones((len(combs), num_per_set))* -1
    for i, comb in enumerate(combs):
        combs_array[i, :len(comb)] = comb
    return combs_array.astype(int)

def compressLabel(label):
    label = label.numpy().astype(int)
    singleton_cnt = np.sum(np.sum(label >= 0, 1) == 1)
    label_dict = {str(e):e for e in range(singleton_cnt)}
    label_compressed = [[e,e] for e in range(singleton_cnt)]
    for i, l in enumerate(label[singleton_cnt:]):
        l = l[:np.sum(l>=0)]
        label_compressed.append([label_dict['_'.join([str(e) for e in l[:-1]])], label_dict[str(l[-1])]])
        label_dict['_'.join([str(e) for e in l])] = i+singleton_cnt
    return torch.tensor(label_compressed)

def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    return torch.norm(D, p=2, dim=2)

def trainphase(f_net, optimizer, batch_size, path='/home/***/data1t_ssd/librispeech_3/train'):
    f_net.train()
    dataloader = getDataloader(path, length = 2, batch_size = batch_size, dataset_length = 100000)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    criterion = nn.MarginRankingLoss(0.1)
    combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))
    combinations3 = torch.tensor([[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [4, 3], [4, 4], [5, 4], [7, 4]])
    for i, data in enumerate(t):
        optimizer.zero_grad()
        test_imgs = data
        test_imgs = test_imgs.float().to(device)
        real_batch_size = test_imgs.size(0)

        test_imgs = test_imgs.contiguous().view(real_batch_size * 30, 199, 32)

        embeddings = f_net(test_imgs).contiguous().view(real_batch_size, 30, -1)

        loss = 0

        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[combinations2.transpose(-2, -1)[1]]
            merged2 = g_net(comb2_a, comb2_b)

            comb3_a = merged2[combinations3.transpose(-2, -1)[0]]
            comb3_b = centroids[combinations3.transpose(-2, -1)[1]]
            merged3 = g_net(comb3_a, comb3_b)

            truth = torch.cat([centroids, merged2, merged3])
            preds = embedding[:-5]
            dists = pairwiseDists(preds, truth)
            losses = torch.zeros(25, device=device)
            for i in range(25):
                if i < 5:
                    weight = 1
                elif i < 15:
                    weight = 0.5
                else:
                    weight = 0.5
                dist = dists[i]
                losses[i] = weight * criterion(dist[[e for e in range(25) if e != i]], dist[[i]*24], torch.ones(24, device=device))
            loss += torch.mean(losses)

        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(f_net, batch_size, path='/home/***/data1t_ssd/librispeech_3/val'):
    f_net.eval()
    dataLoader = getDataloader(path, length = 2, batch_size = batch_size, dataset_length = 1000)
    all_cnt = 0
    hit_cnt = 0
    combinations2 = torch.tensor(list(itertools.combinations(list(range(5)),2)))
    combinations3 = torch.tensor([[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [4, 3], [4, 4], [5, 4], [7, 4]])
    for i, data in enumerate(dataLoader):
        test_imgs = data
        test_imgs = test_imgs.float().to(device)
        real_batch_size = test_imgs.size(0)
        test_imgs = test_imgs.contiguous().view(real_batch_size * 30, 199, 32)

        embeddings = f_net(test_imgs).contiguous().view(real_batch_size, 30, -1)
        for embedding in embeddings:
            centroids = embedding[-5:]
            comb2_a = centroids[combinations2.transpose(-2, -1)[0]]
            comb2_b = centroids[combinations2.transpose(-2, -1)[1]]
            merged2 = g_net(comb2_a, comb2_b)

            comb3_a = merged2[combinations3.transpose(-2, -1)[0]]
            comb3_b = centroids[combinations3.transpose(-2, -1)[1]]
            merged3 = g_net(comb3_a, comb3_b)

            truth = torch.cat([centroids, merged2, merged3])
            preds = embedding[:-5]
            # dists = pairwiseDists(preds, truth)
            dists = torch.matmul(preds, truth.transpose(0, 1))
            _, res = torch.topk(dists, 1)#, largest=False)
            label = torch.tensor(list(range(25)), device=device)
            all_cnt += 25
            hit_cnt += torch.sum(label == res.squeeze()).item()
    return hit_cnt/all_cnt

def testphase(f_net, batch_size, path='/home/***/data1t_ssd/librispeech_3/test'):
    f_net.eval()
    dataloader = getDataloader(path, length = 2, batch_size = batch_size, dataset_length = 1000)
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
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        test_imgs = data
        test_imgs = test_imgs.float().to(device)
        real_batch_size = test_imgs.size(0)

        test_imgs = test_imgs.contiguous().view(30, 199, 32)

        embeddings = f_net(test_imgs).contiguous().view(30, -1)

        centroids = embeddings[-5:]
        comb2_a = centroids[combinations2.transpose(-2, -1)[0]]
        comb2_b = centroids[combinations2.transpose(-2, -1)[1]]
        merged2 = Func.normalize(comb2_a + comb2_b)#g_net(comb2_a, comb2_b)

        comb3_a = merged2[combinations3.transpose(-2, -1)[0]]
        comb3_b = centroids[combinations3.transpose(-2, -1)[1]]
        merged3 = Func.normalize(comb3_a + comb3_b)#g_net(comb3_a, comb3_b)

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
    f_net = Lstm(32).to(device)

    f_net.apply(weight_init)

    if args.load_model:
        checkpoint = torch.load(args.load_model)
        f_net.load_state_dict(checkpoint)
    if args.mode == 'train':
        os.makedirs(args.save_model, exist_ok=True)
        epochs = args.epochs
        optimizer = optim.Adam(list(f_net.parameters()), lr=3e-4)
        for epoch in range(epochs):
            loss = trainphase(f_net, optimizer, 32, args.train_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            with torch.no_grad():
                accu = validationphase(f_net, 32, args.val_dir)
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save(f_net.state_dict(), os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        with torch.no_grad():
            accu = testphase(f_net, 1, args.test_dir)
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