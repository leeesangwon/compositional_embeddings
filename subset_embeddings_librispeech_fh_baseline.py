import os, sys
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from datareader import LibriReader, LibriReadeload
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
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
    combs = [[i, -1] for i in range(5)]
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

def trainphase(fnet, criterion, optimizer, batch_size, path='/home/***/data1t_ssd/librispeech_mikenet/train'):
    fnet.train()
    dataloader = getDataloader(path, batch_size=batch_size)#getDataloader(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length = 2, batch_size = batch_size, dataset_length = 100000)
    criterion = nn.MarginRankingLoss(0.1)
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
        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]
        labels = torch.tensor([-1 for e in range(valid_batch_size)], device=device).float()
        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        loss = criterion(dists[:valid_batch_size], dists[valid_batch_size:], labels)
        # import pdb;pdb.set_trace()
        # pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        # labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        # loss = criterion(pred.squeeze(), torch.tensor(labels, device=device).float())
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(fnet, batch_size, path='/home/***/data1t_ssd/librispeech_mikenet/test'):
    fnet.eval()
    dataloader = getDataloader(path, batch_size=batch_size)#getDataloader(path='/home/***/data1t_ssd/LibriSpeech/testSpker', length = 2, batch_size = batch_size, dataset_length = 10000)

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
        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]
        labels = torch.tensor([-1 for e in range(valid_batch_size)], device=device).float()
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


def main(args):
    f_net = Lstm(32).to(device)
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
            loss = trainphase(f_net, criterion, optimizer, 128, args.train_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            with torch.no_grad():
                accu = validationphase(f_net, 128, args.val_dir)
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save(f_net.state_dict(), os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        with torch.no_grad():
            accu = validationphase(f_net, 128, args.test_dir)
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