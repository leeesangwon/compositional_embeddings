import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import os
from concurrent.futures import ProcessPoolExecutor as PoolExecutor

def AugImg(images):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    seq = iaa.Sequential(
        [   
            iaa.Resize({"height": 64, "width": 64}), 
            # sometimes(iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="edge")),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1), "y": (0.9, 1)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-10, 10),
                shear=(-10, 10),
                order=[0, 1],
                cval=255,
                mode="constant"
            )),

            # sometimes(
            #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
            # ),

            # In some images distort local areas with varying strength.
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), mode='edge'))
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    images_aug = seq(images=images)# > 200)*255
    return images_aug.astype(np.uint8)

def augClass(classname):
    for root, dirs, files in os.walk(os.path.join("data/training", classname)):
        for f in files:
            dest = root.replace('training', 'training_aug')
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in range(50):
                image = imageio.imread(os.path.join(root, f))
                new_images = AugImg(image)
                imageio.imwrite(os.path.join(dest, f.replace('.png', '_{}.png'.format(i))), new_images)

if __name__ == "__main__":
    baseroot = "data/training"
    classes = os.listdir(baseroot)
    with PoolExecutor() as executor:
        executor.map(augClass, classes)