import torch
import numpy as np
import os, random
from torch.utils.data.dataset import Dataset
# import imageio
# from data_aug import AugImg
import itertools
# import cv2
import torchvision.transforms as transforms
# from PIL import Image
# from room_simulation_augmentation import mergeWav
import soundfile as sf
import librosa 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from python_speech_features import mfcc
from tqdm import tqdm
import argparse

class LibriReader_fg(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=100000):
        self.path = path
        self.length = length
        self.audios = {}
        self.speakers = os.listdir(self.path)
        for speaker in self.speakers:
            self.audios[speaker] = []
            speaker_path = os.path.join(self.path, speaker)
            for root, dirs, files in os.walk(speaker_path):
                for f in files:
                    if f.split('.')[-1] == 'flac':
                        self.audios[speaker].append(os.path.join(root, f))#os.listdir(os.path.join(self.path, speaker))

        self.dataset_length = dataset_length
        self.combs2 = list(itertools.combinations(list(range(5)), 2))
        self.combs3 = list(itertools.combinations(list(range(5)), 3))

    def getClip(self, sel_speakers):
        audio, fs = sf.read(random.choice(self.audios[sel_speakers]))
        while len(audio) < self.length*fs:
            audio, fs = sf.read(random.choice(self.audios[sel_speakers]))
        start_t = random.randint(0, int(len(audio) - self.length*fs))
        return audio[start_t:start_t+self.length*fs], fs

    def __getitem__(self, index, volumenorm=False):
        random.shuffle(self.speakers)
        num_speakers = 5
        sel_speakers = self.speakers[:num_speakers]
        sample_clips = []
        ref_clips = []
        for i in range(num_speakers):
            clip, fs = self.getClip(sel_speakers[i])
            sample_clips.append(clip)
            clip, fs = self.getClip(sel_speakers[i])
            ref_clips.append(clip)
        
        for comb in self.combs2:
            sample_a = sample_clips[comb[0]]
            sample_b = sample_clips[comb[1]]
            sample_clips.append(sample_a + sample_b)

        for comb in self.combs3:
            sample_a = sample_clips[comb[0]]
            sample_b = sample_clips[comb[1]]
            sample_c = sample_clips[comb[2]]
            sample_clips.append(sample_a + sample_b + sample_c)

        sample_mfcc = []
        ref_mfcc = []

        for s in sample_clips:
            if volumenorm:
                smax = np.amax(s)
                smin = np.amin(s)
                mean = np.mean(s)
                s = (s-mean)/(smax-smin+1e-6)
            sample_mfcc.append(mfcc(s, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512))
        
        for r in ref_clips:
            if volumenorm:
                smax = np.amax(r)
                smin = np.amin(r)
                mean = np.mean(r)
                r = (r-mean)/(smax-smin+1e-6)
            ref_mfcc.append(mfcc(r, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512))
        
        res_mfcc = np.stack(sample_mfcc+ref_mfcc)

        return sample_clips + ref_clips#res_mfcc

    def __len__(self):
        return self.dataset_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OmniGlot dataset')
    parser.add_argument('input_path',  type=str, help='input path')
    parser.add_argument('output_path',  type=str, help='output path')
    parser.add_argument('number_samples',  type=int, help='numer of samples to generate')
    args = parser.parse_args()

    dataset = LibriReader_fg(path=args.input_path, length=2, dataset_length=args.number_samples)#LibriReader(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', dataset_length=500000)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        feature = data

        dest_path = os.path.join(args.output_path, str(i))
        os.makedirs(dest_path, exist_ok=True)
        for f_cnt, f in enumerate(feature):
            librosa.output.write_wav(os.path.join(dest_path, '{}.wav'.format(f_cnt)), f[0].numpy(), 16000)

        # dest = args.output_path
        # os.makedirs(dest, exist_ok=True)
        # torch.save(feature[0], os.path.join(dest, '{}.pt'.format(i)))