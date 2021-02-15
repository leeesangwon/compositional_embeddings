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
# import librosa 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from python_speech_features import mfcc
from tqdm import tqdm
import argparse

class LibriReader(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=10000):
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
    
    def volumeNorm(self, audio):
        smax = np.amax(audio)
        smin = np.amin(audio)
        mean = np.mean(audio)
        audio = (audio-mean)/(smax-smin+1e-6)
        return audio

    def getClip(self, sel_speakers):
        audio, fs = sf.read(random.choice(self.audios[sel_speakers]))
        while len(audio) < self.length*fs:
            audio, fs = sf.read(random.choice(self.audios[sel_speakers]))
        start_t = random.randint(0, int(len(audio) - self.length*fs))
        return audio[start_t:start_t+self.length*fs], fs

    def __getitem__(self, index, volumenorm=False):
        random.shuffle(self.speakers)
        num_speakers = random.randint(1, 5)
        sel_speakers = self.speakers[:num_speakers + 1]
        audio_clip0, fs = self.getClip(sel_speakers[0])
        merge = audio_clip0
        # speakers = [merge]
        for i in range(1, num_speakers):
            clip, fs = self.getClip(sel_speakers[i])
            # speakers.append(clip)
            merge += clip
        
        

        audio_clip_neg, _ = self.getClip(sel_speakers[-1])
        audio_clip_pos, _ = self.getClip(sel_speakers[0])

        if volumenorm:
            merge = self.volumeNorm(merge)
            audio_clip_neg = self.volumeNorm(audio_clip_neg)
            audio_clip_pos = self.volumeNorm(audio_clip_pos)


        feature_merge = mfcc(merge, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)
        feature_neg = mfcc(audio_clip_neg, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)
        feature_pos = mfcc(audio_clip_pos, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)

        return feature_merge, feature_pos, feature_neg, num_speakers #speakers, merge, audio_clip_pos, audio_clip_neg, fs, num_speakers#

    def __len__(self):
        return self.dataset_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OmniGlot dataset')
    parser.add_argument('input_path',  type=str, help='input path')
    parser.add_argument('output_path',  type=str, help='output path')
    parser.add_argument('number_samples',  type=int, help='numer of samples to generate')
    args = parser.parse_args()

    dataset = LibriReader(path=args.input_path, dataset_length=args.number_samples)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        feature_merge, feature_pos, feature_neg, num_speakers = data
        dest = '{}/{}_{}'.format(args.output_path, i, num_speakers[0].item())
        os.makedirs(dest, exist_ok=True)
        torch.save(feature_merge[0], os.path.join(dest, 'feature_merge.pt'),)
        torch.save(feature_pos[0], os.path.join(dest, 'feature_pos.pt'))
        torch.save(feature_neg[0], os.path.join(dest, 'feature_neg.pt'))