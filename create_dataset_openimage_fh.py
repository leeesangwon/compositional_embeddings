import os, sys
import pickle
import argparse
import imageio
import cv2
import numpy as np 

def crop(img, box):
    h, w = img.shape[0], img.shape[1]
    left, right, top, bottom = int(w*box[0]), int(w*box[1]), int(h*box[2]), int(h*box[3])
    cropimg = img[top:bottom, left:right]
    return cropimg

def resize(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, c = img.shape
    res = np.zeros((max(h, w), max(h, w), c))
    res[:h, :w] = img
    res = cv2.resize(res, (256 , 256))
    return res

def readimg(src, dest, mode, name):
    img_path = os.path.join(src, mode, name)
    img = imageio.imread(img_path)
    img = resize(img)
    os.makedirs('{}/{}_256'.format(dest, mode), exist_ok=True)
    imageio.imwrite(os.path.join('{}/{}_256'.format(dest, mode), name), img.astype(np.uint8))

def subreadimg(src, dest, mode, line):
    data = line.strip().split(',')
    imgid, classname, left, right, top, bottom = data[0], data[2], float(data[4]), float(data[5]), float(data[6]), float(data[7])
    classname = classname.split('/')[-1]
    img = imageio.imread(os.path.join(src, mode, imgid+'.jpg'))
    cropimg = crop(img, [left, right, top, bottom]).astype(np.uint8)
    # cropimg = resize(cropimg).astype(np.uint8)
    os.makedirs('{}/{}_subclass/{}'.format(dest, mode, classname), exist_ok=True)
    imageio.imwrite('{}/{}_subclass/{}/{}.jpg'.format(dest, mode, classname, line.strip().replace('/', '').replace(',', '')), cropimg)

def dumpDict(src, dest, mode):
    img2class = {}
    class2img = {}
    # subclass_info = {}
    with open(os.path.join(src, '{}-annotations-bbox.csv'.format(mode))) as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(',')
            img_id = data[0]
            _class = data[2]
            # left, right, top, bottom = data[4:8]

            # if _class not in subclass_info:
            #     subclass_info[_class] = []
            # subclass_info[_class].append([img_id, left, right, top, bottom]) 

            if img_id not in img2class:
                img2class[img_id] = []
            img2class[img_id].append(_class)

            if _class not in class2img:
                class2img[_class] = []
            class2img[_class].append(img_id)


    # with open('subclass_info.pkl', 'wb') as f:
    #     pickle.dump(subclass_info, f)

    with open(os.path.join(dest, 'img2class_{}.pkl'.format(mode)), 'wb') as f:
        pickle.dump(img2class, f)

    with open(os.path.join(dest, 'class2img_{}.pkl'.format(mode)), 'wb') as f:
        pickle.dump(class2img, f)

def main(args):
    os.makedirs(args.dest, exist_ok=True)
    for mode in ['train', 'validation', 'test']:
        with open(os.path.join(args.src, '{}-annotations-bbox.csv'.format(mode))) as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines[:500]:
                subreadimg(args.src, args.dest, mode, line)

        names = os.listdir(os.path.join(args.src, mode))
        for i, name in enumerate(names):
            readimg(args.src, args.dest, mode, name)
            if i == 100:
                break

        dumpDict(args.src, args.dest, mode)

    with open(os.path.join(args.dest, 'class2img_train.pkl'), 'rb') as f:
        train_data_dict = pickle.load(f)

    with open(os.path.join(args.dest, 'class2img_validation.pkl'), 'rb') as f:
        train_data_dict = pickle.load(f)
    
    with open(os.path.join(args.dest, 'class2img_test.pkl'), 'rb') as f:
        test_data_dict = pickle.load(f)

    class_list = list(train_data_dict.keys())

    with open (os.path.join(args.dest, "train_classes.txt"), 'w') as f:
        for _class in class_list[:500]:
            if len(train_data_dict[_class]) >=100:
                f.write(_class+'\n')

    with open (os.path.join(args.dest, "validation_classes.txt"), 'w') as f:
        for _class in class_list[:500]:
            f.write(_class+'\n')
            
    with open (os.path.join(args.dest, "test_classes.txt"), 'w') as f:
        for _class in list(test_data_dict.keys()):
            if _class not in class_list[:500]:
                f.write(_class+'\n')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('src', type=str,
        help='path of data directory')
    parser.add_argument('dest', type=str,
        help='path of save directory')

    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))