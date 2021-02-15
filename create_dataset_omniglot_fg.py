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

class Omniglot(Dataset):
    def __init__(self, path,num_samples):
        self.root = path
        self.classes = os.listdir(self.root)
        self.class_per_set = 5
        # self.num_per_set = num_per_set
        self.num_per_class = 1
        self.num_samples = num_samples
        # self.combs = []
        # for i in range(1, self.num_per_set+1):
        #     self.combs.extend(list(itertools.combinations(list(range(self.class_per_set)),i)))
        self.combinations = getCombination(5, 3)
        
    def readImg(self, classname, num):
        imgs = np.ones((num, 64, 64))*255
        imgdir = os.path.join(self.root, classname)
        imgnames = random.sample(os.listdir(imgdir), num)
        for i, imgname in enumerate(imgnames):
            img = imageio.imread(os.path.join(imgdir, imgname))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            imgs[i] = img
        return imgs

    def __getitem__(self, index):
        random.shuffle(self.classes)
        images = np.ones((25, 64, 64))*255
        ref_imgs = np.ones((self.class_per_set, 64, 64))*255

        for i, classname in enumerate(self.classes[:self.class_per_set]):
            imgs = self.readImg(classname, self.num_per_class+1)
            images[i*self.num_per_class:i*self.num_per_class+self.num_per_class] = imgs[:-1]
            ref_imgs[i] = imgs[-1]

        
        comb_imgs = np.ones((len(self.combinations), 64, 64))*255
        for comb_cnt, combination in enumerate(self.combinations):
            for idx in combination:
                if idx >= 0:
                    comb_imgs[comb_cnt] = np.minimum(comb_imgs[comb_cnt], AugImg(images[idx]))

        images[5:] = comb_imgs

        background = (np.minimum(np.ones((25, 64, 64)),np.random.normal(0.9, 0.1, (25, 64, 64)))*255).astype(np.uint8)
        images = np.minimum(background, images)

        background = (np.minimum(np.ones((5, 64, 64)),np.random.normal(0.9, 0.1, (5, 64, 64)))*255).astype(np.uint8)
        ref_imgs = np.minimum(background, ref_imgs)
        
        return images, ref_imgs

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OmniGlot dataset')
    parser.add_argument('input_path',  type=str, help='input path')
    parser.add_argument('output_path',  type=str, help='output path')
    parser.add_argument('number_samples',  type=int, help='numer of samples to generate')
    args = parser.parse_args()

    dataset = Omniglot(path = args.input_path, num_samples=args.number_samples) #(path='/home/***/data1t_ssd/LibriSpeech/Spker', length=2, dataset_length=10000)#LibriReader(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', dataset_length=500000)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        test_img, ref_img = data
        test_img = test_img[0].numpy().astype(np.uint8)
        ref_img = ref_img[0].numpy().astype(np.uint8)
        dest = '{}/{}'.format(args.output_path, i)
        os.makedirs(dest, exist_ok=True)
        for img_cnt, img in enumerate(test_img):
            imageio.imwrite(os.path.join(dest, 'test_{}.jpg'.format(img_cnt)), img)
        for img_cnt, img in enumerate(ref_img):
            imageio.imwrite(os.path.join(dest, 'ref_{}.jpg'.format(img_cnt)), img)