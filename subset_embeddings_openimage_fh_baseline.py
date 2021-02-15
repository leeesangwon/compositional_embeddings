import os, sys
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from datareader import OpenImage_reader
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


EMBEDDING_DIM = 128

class F_net (nn.Module):
    def __init__ (self):
        super(F_net, self).__init__()
        self.cnn = models.resnet18(pretrained=True).to(device)
        self.cnn.fc = nn.Linear(512, EMBEDDING_DIM)

    def forward (self, X):
        x = self.cnn(X)
        x = Func.normalize(x)
        return x
        


def getDataloader(imgpath, i2cdictpath, c2idictpath, filelist, subclass, iter_size=10, batch_size=32):
    dataset = OpenImage_reader(imgpath, i2cdictpath, c2idictpath, filelist, subclass, 256, iter_size, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[128, 128, 128],
                                                    std=[127, 127, 127])
                                                    ]))
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return dataloader


def trainphase(fnet, optimizer, batch_size, path):
    fnet.train()
    dataloader = getDataloader(os.path.join(path, 'train_256'), os.path.join(path, 'img2class_train.pkl'), os.path.join(path, 'class2img_train.pkl'), 
    os.path.join(path, 'train_classes.txt'), os.path.join(path, 'train_subclass'), 100, batch_size)
    criterion = nn.MarginRankingLoss(0.1)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))

    for i, data in enumerate(t):
        
        optimizer.zero_grad()
        test_imgs, pos_imgs, neg_imgs = data
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        # imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device).float()
        embeddings = fnet(imgs)

        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]
        labels = torch.tensor([-1 for e in range(valid_batch_size)], device=device).float()
        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        loss = criterion(dists[:valid_batch_size], dists[valid_batch_size:], labels)

        # pred = gnet(torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0), embeddings[valid_batch_size:])
        # labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        # loss = criterion(pred.squeeze(), torch.tensor(labels, device=device).float())
        loss.backward()
        optimizer.step()
    return loss.item()

def validationphase(fnet, batch_size, path, mode):
    fnet.eval()
    if mode == 'validation':
        iter_num = 1
    else:
        iter_num = 100
    dataloader = getDataloader(os.path.join(path, '{}_256'.format(mode)), os.path.join(path, 'img2class_{}.pkl'.format(mode)),os.path.join(path, 'class2img_{}.pkl'.format(mode)), 
    os.path.join(path, '{}_classes.txt'.format(mode)), os.path.join(path, '{}_subclass'.format(mode)), iter_num, batch_size)

    hit_cnt = 0
    all_cnt = 0
    ytrue = []
    ypred = []
    results = {}
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        test_imgs, pos_imgs, neg_imgs = data
        # num_sub_class = torch.sum(catids).item()
        # if num_sub_class not in results:
        #     results[num_sub_class] = [[], []]
        valid_batch_size = test_imgs.size(0)
        imgs = torch.cat([test_imgs, pos_imgs, neg_imgs], 0)
        # imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device).float()
        embeddings = fnet(imgs)

        testembeddings = torch.cat([embeddings[:valid_batch_size], embeddings[:valid_batch_size]], 0)
        refembeddings = embeddings[valid_batch_size:]
        # labels = torch.tensor([-1 for e in range(valid_batch_size)], device=device).float()
        dists = torch.sum((testembeddings - refembeddings)**2, 1)
        labels = [1 for e in range(valid_batch_size)] + [0 for e in range(valid_batch_size)]
        ypred.append(torch.sigmoid(1-dists.squeeze()))
        pred = dists.squeeze() < 0.5
        ytrue += labels
        hit_cnt += torch.sum(pred.float() == torch.tensor(labels, device=device).float()).item()
        all_cnt += valid_batch_size*2
        
    ypred = torch.cat(ypred, 0)
    roc=roc_auc_score(np.array(ytrue), ypred.data.cpu().numpy())
    return hit_cnt/all_cnt, roc#, results

def main(args):
    f_net = F_net().to(device)
    # f_net.apply(weight_init)

    if args.load_model:
        checkpoint = torch.load(args.load_model)
        f_net.load_state_dict(checkpoint)
    if args.mode == 'train':
        os.makedirs(args.save_model, exist_ok=True)
        epochs = args.epochs
        optimizer = optim.Adam(list(f_net.parameters()), lr=3e-4)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            loss = trainphase(f_net, optimizer, 128, args.data_dir)
            print('epoch {}, training loss is {}'.format(epoch, loss))

            torch.manual_seed(0)
            with torch.no_grad():
                accu = validationphase(f_net, 32, args.data_dir, 'validation')
            print('epoch {}, val accu is {}'.format(epoch, accu))

            torch.save(f_net.state_dict(), os.path.join(args.save_model, 'checkpoint_epoch{}.pt'.format(epoch)))
    else:
        torch.manual_seed(0)
        with torch.no_grad():
            accu = validationphase(f_net, 32, args.data_dir, 'test')
        print('test accu is {}'.format(accu))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['train', 'test'],
        help='select to train a new model or test', default='test')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory')
    parser.add_argument('--save_model', type=str, 
        help='Directory Path of saving model checkpoint')
    parser.add_argument('--load_model', type=str, 
        help='Path of loading model checkpoint')
    parser.add_argument('--epochs', type=int,
        help='number of epochs in training')
    
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))