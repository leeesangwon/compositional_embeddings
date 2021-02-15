import numpy as np
import os, sys
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from torch.utils.data.dataset import Dataset
from h_net import *
# from datareader import Omniglot, IdealOmniglot, IdealSeqOmniglot, Omniglot1on1
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from cnn import resnet18
from initialization import weight_init
import random
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
import imageio
import argparse

EMBEDDING_DIM = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=8)
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

def trainphase(fnet, gnet, criterion, optimizer, path='/media/***/adatassd/omniglot_mike/train'):
    fnet.train()
    gnet.train()
    dataloader = getDataloader(path)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        
        optimizer.zero_grad()
        test_imgs, pos_imgs, neg_imgs, _ = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256

        embeddings = fnet(imgs)
        # embeddings = embeddings.contiguous().view(valid_batch_size, 3, 32)
        pred = gnet(torch.cat([embeddings[: valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        loss = criterion(pred.squeeze(), torch.tensor(labels, device=device).float())
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(fnet, gnet, path='/media/***/adatassd/omniglot_mike/val'):
    torch.manual_seed(0)
    fnet.eval()
    gnet.eval()
    dataLoader = getDataloader(path)
    hit_cnt = 0
    all_cnt = 0
    for i, data in enumerate(dataLoader):
        test_imgs, pos_imgs, neg_imgs, _ = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256

        embeddings = fnet(imgs)

        pred = gnet(torch.cat([embeddings[0: valid_batch_size], embeddings[0:valid_batch_size]], 0), embeddings[valid_batch_size:])
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]

        pred = pred.squeeze() > 0.5

        hit_cnt += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt += valid_batch_size*2
        # import pdb;pdb.set_trace()
    return hit_cnt/all_cnt

def testphase(fnet, gnet, path='/media/***/adatassd/omniglot_mike/test'):
    torch.manual_seed(0)
    fnet.eval()
    gnet.eval()
    dataLoader = getDataloader(path)
    hit_cnt = 0
    all_cnt = 0
    y_true = []
    y_scores = []
    for i, data in enumerate(dataLoader):
        test_imgs, pos_imgs, neg_imgs, _ = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        imgs = imgs.to(device).float()/256

        embeddings = fnet(imgs)

        pred = gnet(torch.cat([embeddings[0: valid_batch_size], embeddings[0:valid_batch_size]], 0), embeddings[valid_batch_size:])
        y_scores.append(pred)
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        y_true += labels
        pred = pred.squeeze() > 0.5

        hit_cnt += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt += valid_batch_size*2
    y_scores = torch.cat(y_scores, 0)
    roc=roc_auc_score(np.array(y_true), y_scores.data.cpu().numpy())
    return hit_cnt/all_cnt, roc


def main(args):
    f_net = F_net().to(device)
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
            loss = trainphase(f_net, mike_net, criterion, optimizer, args.train_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            with torch.no_grad():
                accu = validationphase(f_net, mike_net, args.val_dir)
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save({'f_net':f_net.state_dict(), 'g_net':mike_net.state_dict()}, os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        with torch.no_grad():
            accu = testphase(f_net, mike_net, args.test_dir)
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