#%%

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
import soundfile as sf
import librosa
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from python_speech_features import mfcc
from tqdm import tqdm
import pickle
# import matplotlib.pyplot as plt

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


class Omniglot(Dataset):
    def __init__(self, path, class_per_set, num_per_set, num_per_class,num_samples):
        self.root = path
        self.classes = os.listdir(self.root)
        self.class_per_set = class_per_set
        self.num_per_set = num_per_set
        self.num_per_class = num_per_class
        self.num_samples = num_samples
        self.combs = []
        for i in range(1, self.num_per_set+1):
            self.combs.extend(list(itertools.combinations(list(range(self.class_per_set)),i)))
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
        return self.num_samples#len(self.classes)//self.class_per_set

class IdealOmniglot(Dataset):
    def __init__(self, path, class_per_set, num_per_set, num_per_class,num_samples):
        self.root = path
        self.classes = os.listdir(self.root)
        self.class_per_set = class_per_set
        self.num_per_set = num_per_set
        self.num_per_class = num_per_class
        self.num_samples = num_samples
        self.combs = []
        for i in range(1, self.num_per_set+1):
            self.combs.extend(list(itertools.combinations(list(range(self.class_per_set)),i)))
        
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
        images = np.ones((15, 64, 64))*255
        ref_imgs = np.ones((15, 64, 64))*255

        for i, classname in enumerate(self.classes[:self.class_per_set]):
            imgs = self.readImg(classname, self.num_per_class+1)
            images[i] = imgs[:-1]
            ref_imgs[i] = imgs[-1]

        combinations = getCombination(5, 2)
        comb_imgs = np.ones((len(combinations), 64, 64))*255
        comb_ref_imgs = np.ones((len(combinations), 64, 64))*255
        for comb_cnt, combination in enumerate(combinations):
            for idx in combination:
                comb_imgs[comb_cnt] = np.minimum(comb_imgs[comb_cnt], AugImg(images[idx]))
                comb_ref_imgs[comb_cnt] = np.minimum(comb_ref_imgs[comb_cnt], AugImg(ref_imgs[idx]))

        images[5:] = comb_imgs
        ref_imgs[5:] = comb_ref_imgs

        background = (np.minimum(np.ones((15, 64, 64)),np.random.normal(0.9, 0.1, (15, 64, 64)))*256).astype(np.uint8)
        images = np.minimum(background, images)

        background = (np.minimum(np.ones((15, 64, 64)),np.random.normal(0.9, 0.1, (15, 64, 64)))*256).astype(np.uint8)
        ref_imgs = np.minimum(background, ref_imgs)
        
        return images, ref_imgs

    def __len__(self):
        return self.num_samples#len(self.classes)//self.class_per_set

class omniglot_load(Dataset):
    def __init__(self, path):
        self.path = path
        self.samples = os.listdir(self.path)
    
    def __getitem__(self, index):
        test_img = np.zeros((25, 64, 64))
        ref_img = np.zeros((5, 64, 64))
        sample = self.samples[index]
        root = os.path.join(self.path, sample)
        for i in range(25):
            test_img[i] = imageio.imread(os.path.join(root, 'test_{}.jpg'.format(i)))
        for i in range(5):
            ref_img[i] = imageio.imread(os.path.join(root, 'ref_{}.jpg'.format(i)))
        return test_img, ref_img

    def __len__(self):
        return len(self.samples)


class SeqOmniglot(Dataset):
    def __init__(self, path):
        self.root = path
        self.samples = os.listdir(self.root)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = np.ones((15, 64, 64))*255
        ref_imgs = np.ones((5, 64, 64))*255
        for i in range(15):
            img = imageio.imread(os.path.join(self.root, sample, 'test_{}.png'.format(i)))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            images[i] = img
        for i in range(5):
            img = imageio.imread(os.path.join(self.root, sample, 'ref_{}.png'.format(i)))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            ref_imgs[i] = img

        return images, ref_imgs

    def __len__(self):
        return len(self.samples)

class IdealSeqOmniglot(Dataset):
    def __init__(self, path):
        self.root = path
        self.samples = os.listdir(self.root)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = np.ones((15, 64, 64))*255
        ref_imgs = np.ones((15, 64, 64))*255
        for i in range(15):
            img = imageio.imread(os.path.join(self.root, sample, 'test_{}.png'.format(i)))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            images[i] = img

        combinations = getCombination(5, 2)
        for i in range(5):
            img = imageio.imread(os.path.join(self.root, sample, 'ref_{}.png'.format(i)))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            ref_imgs[i] = img

        for comb_cnt, combination in enumerate(combinations):
            for idx in combination:
                ref_imgs[comb_cnt+5] = np.minimum(ref_imgs[comb_cnt+5], ref_imgs[idx])

        return images, ref_imgs

    def __len__(self):
        return len(self.samples)

class Omniglot1on1(Dataset):
    def __init__(self, path, num):
        self.root = path
        self.samples = os.listdir(self.root)
        self.num_samples = num
        self.imglist = {}
        for sample in self.samples:
            self.imglist[sample] = os.listdir(os.path.join(self.root, sample))

    
    def readImg(self, classname, num):
        imgs = np.ones((num, 64, 64))*255
        imgnames = random.sample(self.imglist[classname], num)
        for i, imgname in enumerate(imgnames):
            img = imageio.imread(os.path.join(self.root, classname, imgname))
            if img.shape[0] != 64:
                img = cv2.resize(img, (64, 64))
            imgs[i] = img
        return imgs

    def __getitem__(self, index):
        samples = random.sample(self.samples, 3)
        images = np.ones((6, 64, 64))*255
        res = np.ones((3, 64, 64))*255

        for i in range(3):
            images[i*2:i*2+2] = self.readImg(samples[i], 2)
        res[0] = np.minimum(images[0], images[2])
        postype = random.randint(0, 1)
        if postype:
            res[1] = np.minimum(images[1], images[3])
        else:
            res[1] = images[1]
        # res[1] = images[1]

        negtype = random.randint(0, 1)
        if negtype:
            res[2] = np.minimum(images[1], images[4])
        else:
            res[2] = images[4]
        # res[2] = images[4]
        background = (np.minimum(np.ones((3, 64, 64)),np.random.normal(0.9, 0.1, (3, 64, 64)))*256).astype(np.uint8)
        res = np.minimum(background, res)
        return res

    def __len__(self):
        return self.num_samples

class Omniglot1on1Aug(Dataset):
    def __init__(self, path, num):
        self.root = path
        self.samples = os.listdir(self.root)
        self.num_samples = num
        self.imglist = {}
        for sample in self.samples:
            self.imglist[sample] = os.listdir(os.path.join(self.root, sample))

    
    def readImg(self, classname, num):
        imgs = np.ones((num, 64, 64))*255
        imgnames = random.sample(self.imglist[classname], num)
        for i, imgname in enumerate(imgnames):
            img = imageio.imread(os.path.join(self.root, classname, imgname))
            # if img.shape[0] != 64:
            #     img = cv2.resize(img, (64, 64))
            imgs[i] = AugImg(img)
        return imgs

    def __getitem__(self, index):
        samples = random.sample(self.samples, 3)
        images = np.ones((6, 64, 64))*255
        res = np.ones((3, 64, 64))*255

        for i in range(3):
            images[i*2:i*2+2] = self.readImg(samples[i], 2)
        res[0] = np.minimum(images[0], images[2])
        postype = random.randint(0, 1)
        if postype:
            res[1] = np.minimum(images[1], images[3])
        else:
            res[1] = images[1]
        # res[1] = images[1]

        negtype = random.randint(0, 1)
        if negtype:
            res[2] = np.minimum(images[1], images[4])
        else:
            res[2] = images[4]
        # res[2] = images[4]
        background = (np.minimum(np.ones((3, 64, 64)),np.random.normal(0.9, 0.1, (3, 64, 64)))*256).astype(np.uint8)
        res = np.minimum(background, res)
        return res

    def __len__(self):
        return self.num_samples


class CocoReader(Dataset):
    def __init__(self, path='/home/***/coco/data', dataset='train2017', filelist = 'filtered',img_size = 128, transform=None):
        self.root = path
        self.dataset = dataset
        self.img_size = img_size
        with open(os.path.join(path, '{}_img_cats_{}.txt'.format(dataset, filelist))) as f:
            self.lines = f.readlines()
        self.catids = os.listdir(os.path.join(path, '{}_processed_large_resize'.format(dataset)))
        self.catid_map = {}
        for i, catid in enumerate(self.catids):
            self.catid_map[catid] = i
        self.subimgnames = {}
        for catid in self.catids:
            self.subimgnames[catid] = os.listdir(os.path.join(path, '{}_processed_large_resize'.format(dataset), catid))
        self.transform = transform

    def resize(self, img):
        # print(img.shape)
        # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, c = img.shape
        res = np.zeros((max(h, w), max(h, w), c))
        res[:h, :w] = img
        res = cv2.resize(res, (self.img_size , self.img_size ))
        return res

    def __getitem__(self, index):
        datainfo = self.lines[index]
        datainfo = datainfo.strip().split()
        img_name = datainfo[0]
        val_catids = datainfo[1:]
        random.shuffle(self.catids)
        label = np.zeros(80)
        for catid in val_catids:
            label[self.catid_map[catid]] = 1
        for catid in self.catids:
            if catid not in val_catids:
                negid = catid
                break
        posid = random.choice(val_catids)
        img = imageio.imread(os.path.join(self.root, 'images', '{}r'.format(self.dataset), img_name))


        pos_dir = os.path.join(self.root, '{}_processed_large_resize'.format(self.dataset), posid)
        posimg = imageio.imread(os.path.join(pos_dir, random.choice(self.subimgnames[posid])))


        neg_dir = os.path.join(self.root, '{}_processed_large_resize'.format(self.dataset), negid)
        negimg = imageio.imread(os.path.join(neg_dir, random.choice(self.subimgnames[negid])))

        if self.transform:
            img = self.transform(img)
            posimg = self.transform(posimg)
            negimg = self.transform(negimg)
        return img, posimg, negimg, label

    def __len__(self):
        return len(self.lines)

class OpenImage_reader_classification(Dataset):
    def __init__(self, img_path, i2cdictpath, c2idictpath, filelist, subclass, img_size, iters, transform):
        self.img_size = img_size
        self.iters = iters
        self.transform = transform
        self.subclass_path = subclass

        # with open(c2idictpath, 'rb') as f:
        #     self.class2img = pickle.load(f)

        # with open(i2cdictpath, 'rb') as f:
        #     self.img2class = pickle.load(f)
        
        self.subclass = {}
        subclasses = os.listdir(subclass)
        for i in subclasses:
            if i not in self.subclass:
                self.subclass[i] = os.listdir(os.path.join(subclass, i))

        self.classes = []
        with open(filelist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().split('/')[-1] in self.subclass:
                    self.classes.append(line.strip())

        self.img_path = img_path

    def __getitem__(self, index):
        test_class = random.sample(self.classes, 2)

        classname = test_class[0].split('/')[-1]
        posname = random.choice(self.subclass[classname])
        img = imageio.imread(os.path.join(self.subclass_path, classname, posname))

        classname = test_class[0].split('/')[-1]
        posname = random.choice(self.subclass[classname])
        pos_img = imageio.imread(os.path.join(self.subclass_path, classname, posname))

        classname = test_class[1].split('/')[-1]
        negname = random.choice(self.subclass[classname])
        neg_img = imageio.imread(os.path.join(self.subclass_path, classname, negname))

        if self.transform:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img

    def __len__(self):
        return len(self.classes)*self.iters

class OpenImage_reader(Dataset):
    def __init__(self, img_path, i2cdictpath, c2idictpath, filelist, subclass, img_size, iters, transform):
        self.img_size = img_size
        self.iters = iters
        self.transform = transform
        self.subclass_path = subclass

        with open(c2idictpath, 'rb') as f:
            self.class2img = pickle.load(f)

        with open(i2cdictpath, 'rb') as f:
            self.img2class = pickle.load(f)
        
        self.subclass = {}
        subclasses = os.listdir(subclass)
        for i in subclasses:
            if i not in self.subclass:
                self.subclass[i] = os.listdir(os.path.join(subclass, i))

        self.classes = []
        with open(filelist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().split('/')[-1] in self.subclass:
                    self.classes.append(line.strip())

        self.img_path = img_path

    def __getitem__(self, index):
        pos_class = self.classes[index%len(self.classes)]
        imgname = random.choice(self.class2img[pos_class])
        pos_classes = self.img2class[imgname]
        neg_class = random.choice(self.classes)
        while neg_class in pos_classes:
            neg_class = random.choice(self.classes)

        img = imageio.imread(os.path.join(self.img_path, imgname+'.jpg'))

        classname = pos_class.split('/')[-1]
        posname = random.choice(self.subclass[classname])
        pos_img = imageio.imread(os.path.join(self.subclass_path, classname, posname))

        classname = neg_class.split('/')[-1]
        negname = random.choice(self.subclass[classname])
        neg_img = imageio.imread(os.path.join(self.subclass_path, classname, negname))

        if self.transform:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, pos_img, neg_img, 

    def __len__(self):
        return len(self.classes)*self.iters

class CocoReader_label_only(Dataset):
    def __init__(self, path='/home/***/coco/data', dataset='train2017', filelist = 'filtered',img_size = 128, transform=None):
        self.root = path
        self.dataset = dataset
        self.img_size = img_size
        with open(os.path.join(path, '{}_img_cats_{}.txt'.format(dataset, filelist))) as f:
            self.lines = f.readlines()
        self.catids = os.listdir(os.path.join(path, '{}_processed_large_resize'.format(dataset)))
        self.catid_map = {}
        for i, catid in enumerate(self.catids):
            self.catid_map[catid] = i
        self.subimgnames = {}
        for catid in self.catids:
            self.subimgnames[catid] = os.listdir(os.path.join(path, '{}_processed_large_resize'.format(dataset), catid))
        self.transform = transform

    def resize(self, img):
        # print(img.shape)
        # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, c = img.shape
        res = np.zeros((max(h, w), max(h, w), c))
        res[:h, :w] = img
        res = cv2.resize(res, (self.img_size , self.img_size ))
        return res

    def __getitem__(self, index):
        datainfo = self.lines[index]
        datainfo = datainfo.strip().split()
        img_name = datainfo[0]
        val_catids = datainfo[1:]
        random.shuffle(self.catids)
        label = np.zeros(80)
        for catid in val_catids:
            label[self.catid_map[catid]] = 1
        for catid in self.catids:
            if catid not in val_catids:
                negid = catid
                break
        posid = random.choice(val_catids)
        img = imageio.imread(os.path.join(self.root, 'images', '{}r'.format(self.dataset), img_name))

        neg_label = self.catid_map[negid]

        pos_label = self.catid_map[posid]

        neg_label_one_hot = np.zeros(80)
        pos_label_one_hot = np.zeros(80)
        neg_label_one_hot[neg_label] = 1
        pos_label_one_hot[pos_label] = 1


        # pos_dir = os.path.join(self.root, '{}_processed_large_resize'.format(self.dataset), posid)
        # posimg = imageio.imread(os.path.join(pos_dir, random.choice(self.subimgnames[posid])))


        # neg_dir = os.path.join(self.root, '{}_processed_large_resize'.format(self.dataset), negid)
        # negimg = imageio.imread(os.path.join(neg_dir, random.choice(self.subimgnames[negid])))

        if self.transform:
            img = self.transform(img)
        #     posimg = self.transform(posimg)
        #     negimg = self.transform(negimg)
        return img, pos_label, neg_label#pos_label_one_hot, neg_label_one_hot

    def __len__(self):
        return len(self.lines)

class OpenImage_reader_traverse(Dataset):
    def __init__(self, img_path, i2cdictpath, c2idictpath, filelist, subclass, img_size, iters, transform):
        self.img_size = img_size
        self.iters = iters
        self.transform = transform
        self.subclass_path = subclass

        with open(c2idictpath, 'rb') as f:
            self.class2img = pickle.load(f)

        with open(i2cdictpath, 'rb') as f:
            self.img2class = pickle.load(f)
        
        self.subclass = {}
        subclasses = os.listdir(subclass)
        for i in subclasses:
            if i not in self.subclass:
                self.subclass[i] = os.listdir(os.path.join(subclass, i))

        self.classes = []
        with open(filelist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().split('/')[-1] in self.subclass:
                    self.classes.append(line.strip())

        self.img_path = img_path

    def reshape(self, img):
        h, w, _ = img.shape
        res = np.zeros((max(w, h), max(w, h), 3))
        res[:h, :w] = img
        res = cv2.resize(res, (256, 256))
        return res

    def __getitem__(self, index):
        pos_class = self.classes[index%len(self.classes)]
        imgname = random.choice(self.class2img[pos_class])
        img = imageio.imread(os.path.join(self.img_path, imgname+'.jpg'))
        while len(img.shape) < 3:
            imgname = random.choice(self.class2img[pos_class])
            img = imageio.imread(os.path.join(self.img_path, imgname+'.jpg'))
        height, width, _ = img.shape


        pos_classes = self.img2class[imgname]
        neg_class = random.choice(self.classes)
        while neg_class in pos_classes:
            neg_class = random.choice(self.classes)

        
        # cv2.imwrite('/home/lizeqian/samples/{}.jpg'.format(imgname), img)
        # print(imgname)

        
        grid = 4
        size = max(height, width)//grid
        step = max(height, width)//(grid)
        reses = []
        for i in range(math.ceil(width/step)):
            for j in range(math.ceil(height/step)):
                shape_w = (i+1)*size
                shape_h = (j+1)*size
                for m in range(grid - i):
                    for n in range(grid - j):
                        region = img[n*step:n*step+shape_h,m*step:m*step+shape_w]
                        res = self.reshape(region)
                        
                        # cv2.imwrite('/home/lizeqian/samples/img_{}_{}_{}_{}.jpg'.format(m*step,n*step,m*step+shape_w, n*step+shape_h), res)
                        res = self.transform(res)
                        reses.append(res)

        classname = pos_class.split('/')[-1]
        posname = random.choice(self.subclass[classname])
        pos_img = imageio.imread(os.path.join(self.subclass_path, classname, posname))

        # posname = random.choice(self.subclass[classname])
        # pos_dummy = self.transform(imageio.imread(os.path.join(self.subclass_path, classname, posname)))
        # cv2.imwrite('/home/lizeqian/samples/{}_{}_pos.jpg'.format(imgname, classname), pos_img)

        classname = neg_class.split('/')[-1]
        negname = random.choice(self.subclass[classname])
        neg_img = imageio.imread(os.path.join(self.subclass_path, classname, negname))
        # cv2.imwrite('/home/lizeqian/samples/{}_{}_neg.jpg'.format(imgname, classname), neg_img)

        if self.transform:
            # img = self.transform(img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        # reses.append(pos_dummy.double())
        reses = torch.stack(reses)
        return reses, pos_img, neg_img, imgname

    def __len__(self):
        return len(self.classes)*self.iters

class LibriReader(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=10000):
        self.path = path
        self.length = length
        self.audios = {}
        self.speakers = os.listdir(self.path)
        for speaker in self.speakers:
            self.audios[speaker] = os.listdir(os.path.join(self.path, speaker))
        
        self.mean = np.load('mean.npy')
        self.std = np.load('std.npy')
        self.dataset_length = dataset_length

    def getClip(self, sel_speakers):
        audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
        while len(audio) < self.length*fs:
            audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
        start_t = random.randint(0, int(len(audio) - self.length*fs))
        return audio[start_t:start_t+self.length*fs], fs

    def __getitem__(self, index):
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

        feature_merge = mfcc(merge, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)
        feature_neg = mfcc(audio_clip_neg, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)
        feature_pos = mfcc(audio_clip_pos, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512)

        return feature_merge, feature_pos, feature_neg, num_speakers #speakers, merge, audio_clip_pos, audio_clip_neg, fs, num_speakers#

    def __len__(self):
        return self.dataset_length

class LibriReadeload(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/librispeech_mikenet/train'):
        self.path = path
        self.samples = os.listdir(path)

    def __getitem__(self, index):
        sel_sample = self.samples[index]
        num_speakers = int(sel_sample.split('_')[-1])
        feature_merge = torch.load(os.path.join(self.path, sel_sample, 'feature_merge.pt'))
        feature_pos = torch.load(os.path.join(self.path, sel_sample, 'feature_pos.pt'))
        feature_neg = torch.load(os.path.join(self.path, sel_sample, 'feature_neg.pt'))
        
        return feature_merge, feature_pos, feature_neg, num_speakers #speakers, merge, audio_clip_pos, audio_clip_neg, fs, num_speakers#

    def __len__(self):
        return len(self.samples)

class LibriReader_fg(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=100000):
        self.path = path
        self.length = length
        self.audios = {}
        self.speakers = os.listdir(self.path)
        for speaker in self.speakers:
            self.audios[speaker] = os.listdir(os.path.join(self.path, speaker))
        
        # self.mean = np.load('mean.npy')
        # self.std = np.load('std.npy')
        self.dataset_length = dataset_length
        self.combs2 = list(itertools.combinations(list(range(5)), 2))
        self.combs3 = list(itertools.combinations(list(range(5)), 3))

    def getClip(self, sel_speakers):
        audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
        while len(audio) < self.length*fs:
            audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
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

        return res_mfcc

    def __len__(self):
        return self.dataset_length

class LibriReader_fg_baseline(Dataset):
    def __init__(self, path='/home/***/data1t_ssd/LibriSpeech/trainSpker', length=2, dataset_length=10000):
        self.path = path
        self.length = length
        self.audios = {}
        self.speakers = os.listdir(self.path)
        for speaker in self.speakers:
            self.audios[speaker] = os.listdir(os.path.join(self.path, speaker))
        
        # self.mean = np.load('mean.npy')
        # self.std = np.load('std.npy')
        self.dataset_length = dataset_length
        # self.combs = list(itertools.combinations(list(range(5)), 2))

    def getClip(self, sel_speakers):
        audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
        while len(audio) < self.length*fs:
            audio, fs = sf.read(os.path.join(self.path, sel_speakers, random.choice(self.audios[sel_speakers])))
        start_t = random.randint(0, int(len(audio) - self.length*fs))
        return audio[start_t:start_t+self.length*fs], fs

    def __getitem__(self, index):
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

        sample_mfcc = []
        ref_mfcc = []

        for s in sample_clips:
            sample_mfcc.append(mfcc(s, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512))
        
        for r in ref_clips:
            ref_mfcc.append(mfcc(r, fs, winlen=0.025, winstep=0.01, numcep = 32, nfilt = 40, nfft=512))
        
        res_mfcc = np.stack(sample_mfcc+ref_mfcc)

        return res_mfcc

    def __len__(self):
        return self.dataset_length

class MikeReader(Dataset):
    def __init__(self, path='train'):
        self.src = path #os.path.join('data/mikenet', path)
        self.samples = os.listdir(self.src)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        res = np.zeros((3, 64, 64))
        for i in range(3):
            res[i] = imageio.imread(os.path.join(self.src, sample, '{}.jpg'.format(i)))
        return res
    
    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    # dataset = OmniGlot_fh('data/test', 10000)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8)
    # t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    # for i, data in enumerate(t):
    #     merge, pos, neg, num = data
    #     dest = '/media/***/adatassd/omniglot_mike/test/{}_{}'.format(i, num[0].item())
    #     os.makedirs(dest, exist_ok=True)
    #     imageio.imwrite(os.path.join(dest, 'merge.jpg'), merge[0].numpy().astype(np.uint8))
    #     imageio.imwrite(os.path.join(dest, 'pos.jpg'), pos[0].numpy().astype(np.uint8))
    #     imageio.imwrite(os.path.join(dest, 'neg.jpg'), neg[0].numpy().astype(np.uint8))
    # import pdb;pdb.set_trace()

    # dataset = Omniglot( path = 'data/test',class_per_set=5, num_per_set=3, num_per_class=1, num_samples=1000) #(path='/home/***/data1t_ssd/LibriSpeech/Spker', length=2, dataset_length=10000)#LibriReader(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', dataset_length=500000)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8)
    # t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    # for i, data in enumerate(t):
    #     test_img, ref_img = data
    #     test_img = test_img[0].numpy().astype(np.uint8)
    #     ref_img = ref_img[0].numpy().astype(np.uint8)
    #     dest = '/home/***/subset_embeddings/data/test_aug_3/{}'.format(i)
    #     os.makedirs(dest, exist_ok=True)
    #     for img_cnt, img in enumerate(test_img):
    #         imageio.imwrite(os.path.join(dest, 'test_{}.jpg'.format(img_cnt)), img)
    #     for img_cnt, img in enumerate(ref_img):
    #         imageio.imwrite(os.path.join(dest, 'ref_{}.jpg'.format(img_cnt)), img)

    # dataset = OpenImage_reader('/home/***/data1t_ssd/openimage/test_256', '/home/***/data4t/openimage/class2img_test.pkl', '/home/***/data4t/openimage/test_classes.txt', 
    # '/home/***/data1t_ssd/openimage/test_subclass', 256, 10, transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[128, 128, 128],
    #                                                 std=[127, 127, 127])
    #                                                 ]))
    # for i in range(dataset.__len__()):
    #     print(i)
    #     imgs = dataset.__getitem__(i)
    #     imageio.imwrite('test0.jpg', imgs[0])
    #     imageio.imwrite('test1.jpg', imgs[1])
    #     imageio.imwrite('test2.jpg', imgs[2])
    #     break


    # dataset = MikeReader()
    # for i in range(dataset.__len__()):
    #     print(i)
    #     imgs = dataset.__getitem__(i)


    # dataset = Omniglot1on1('data/training_aug', 500000)
    # for i in range(dataset.__len__()):
    #     dest = os.path.join('data/mikenet/train', str(i))
    #     os.makedirs(dest, exist_ok=True)
    #     imgs = dataset.__getitem__(i)
    #     for cnt, img in enumerate(imgs):
    #         imageio.imwrite(os.path.join(dest, '{}.jpg'.format(cnt)), img.astype(np.uint8))


    # dataset = CocoReader(transform=transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[128, 128, 128],
    #                                                 std=[127, 127, 127])
    #                         ]))
    # dataset.__getitem__(0)
    
    dataset = LibriReader_fg(path='/home/***/data1t_ssd/LibriSpeech/testSpker', length=2, dataset_length=10000)#LibriReader(path='/home/***/data1t_ssd/LibriSpeech/trainSpker', dataset_length=500000)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8)
    t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
    for i, data in enumerate(t):
        feature = data
        dest = '/home/***/data1t_ssd/librispeech_3/test'
        os.makedirs(dest, exist_ok=True)
        torch.save(feature[0], os.path.join(dest, '{}.pt'.format(i)))

        # feature_merge, feature_pos, feature_neg, num_speakers = data
        # dest = '/home/***/data1t_ssd/librispeech_mikenet/train/{}_{}'.format(i, num_speakers[0].item())
        # os.makedirs(dest, exist_ok=True)
        # torch.save(feature_merge[0], os.path.join(dest, 'feature_merge.pt'),)
        # torch.save(feature_pos[0], os.path.join(dest, 'feature_pos.pt'))
        # torch.save(feature_neg[0], os.path.join(dest, 'feature_neg.pt'))

    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, batch_size=50, sampler=sampler, num_workers=0)
    # import time
    # start_time = time.time()
    # for j in range(10):
    #     print(j)
    #     for i, data in enumerate(dataloader):
    #         a = data
    #         print(a[0].size())
    # elapsed_time = time.time() - start_time
    # print(elapsed_time)

# %%
