import torch
import numpy as np
import os, random
from torch.utils.data.dataset import Dataset
import imageio
from data_aug import AugImg
import itertools
import cv2
import torchvision.transforms as transforms
# from PIL import Image
# from room_simulation_augmentation import mergeWav
# import soundfile as sf
# import librosa 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# from python_speech_features import mfcc
from tqdm import tqdm
import argparse

def getCombination(num_per_classes=5, num_per_set=2):
    combs = []
    for i in range(2, num_per_set+1):
        combs.extend(list(itertools.combinations(list(range(num_per_classes)),i)))
    combs_array = np.ones((len(combs), num_per_set))* -1
    for i, comb in enumerate(combs):
        combs_array[i, :len(comb)] = comb
    return combs_array.astype(int)

class OmniGlot_fh(Dataset):
    def __init__(self, path, num):
        self.root = path
        self.classes = os.listdir(self.root)
        self.num_samples = num

    def readImg(self, classname):
        imgdir = os.path.join(self.root, classname)
        imgname = random.choice(os.listdir(imgdir))
        img = imageio.imread(os.path.join(imgdir, imgname))
        if img.shape[0] != 64:
            img = cv2.resize(img, (64, 64))
        return img
    
    def __getitem__(self, index):
        random.shuffle(self.classes)
        num_classes = random.randint(1, 5)
        img = np.zeros((num_classes+2, 64, 64))
        for i, _class in enumerate(self.classes[:num_classes]):
            img[i] = AugImg(self.readImg(_class))
        img[num_classes] = self.readImg(self.classes[0])
        img[num_classes+1] = self.readImg(self.classes[num_classes])
        merge = np.amin(img[:num_classes], axis = 0)
        imgs = np.stack([merge, img[num_classes], img[num_classes+1]])
        background = (np.minimum(np.ones((3, 64, 64)),np.random.normal(0.9, 0.1, (3, 64, 64)))*255).astype(np.uint8)
        imgs = np.minimum(imgs, background)
        return imgs[0], imgs[1], imgs[2], num_classes
        

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OmniGlot dataset')
    parser.add_argument('input_path',  type=str, help='input path')
    parser.add_argument('output_path',  type=str, help='output path')
    parser.add_argument('number_samples',  type=int, help='numer of samples to generate')
    args = parser.parse_args()

    dataset = OmniGlot_fh(args.input_path, args.number_samples)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        merge, pos, neg, num = data
        dest = '{}/{}_{}'.format(args.output_path, i, num[0].item())
        os.makedirs(dest, exist_ok=True)
        imageio.imwrite(os.path.join(dest, 'merge.jpg'), merge[0].numpy().astype(np.uint8))
        imageio.imwrite(os.path.join(dest, 'pos.jpg'), pos[0].numpy().astype(np.uint8))
        imageio.imwrite(os.path.join(dest, 'neg.jpg'), neg[0].numpy().astype(np.uint8))