# Requirement
PyTorch 1.3
imgaug 0.3
opencv 4.2
imageio 2.6
pysoundfile 0.10.2
librosa 0.7.2
python_speech_features 0.6.1

*My environment dependencies are included in environment.yml

# Data generation
Experiment 1:
Download dataset from https://github.com/brendenlake/omniglot
Run this script 3 times for training set, validation set and test set
python create_dataset_omniglot_fg.py <input_dir> <output_dir> <number of samples> 


Experiment 2:
Run this script 3 times for training set, validation set and test set
python create_dataset_omniglot_fh.py <input_dir> <output_dir> <number of samples> 

Experiment 3:
Download Subset with Bounding Boxes from https://storage.googleapis.com/openimages/web/download.html
Merge all data from train_* into one directory train
Run the script once
python create_dataset_openimage_fh.py <input_dir> <output_dir>
*Note: for baseline SlideWin, original test set should be put in <output_dir>

Experiment 4:
The script to process COCO is modified from previous script. Only difference is that the training set and test set contain the same classes.

# Training and testing
Experiment 1:
training:
python subset_embeddings_omniglot_fg.py train <select g()> --train_dir <training set path> --val_dir <validation set path> --save_model <path to save checkpoint> --epochs <number of epochs>

testing:
python subset_embeddings_omniglot_fg.py test <select g()> --test_dir <test set path> --load_model <path to load checkpoint>

TradEmb training:
python subset_embeddings_omniglot_fg_baseline.py train --train_dir <training set path> --val_dir <validation set path> --save_model <path to save checkpoint> --epochs <number of epochs>

TradEmb testing:
python subset_embeddings_omniglot_fg_baseline.py test --test_dir <test set path> --load_model <path to load checkpoint>

Other experiments follow the same way. Codes for those experiments are:

Experiment 2:
subset_embeddings_omniglot_fh.py
subset_embeddings_omniglot_fh_baseline.py


Experiment 3:
subset_embeddings_openimage_fh.py
subset_embeddings_openimage_fh_baseline.py

Experiment 4 uses the same code as Experiment 3, except some minor modifications in dataloader.

Checkpoint files are stored in URL: https://drive.google.com/open?id=1L3rNK1FXcjN5Toc5dq-BzwV-3hnAYiaV. Note that the Google Drive username is anonymized (“bunny bear”) to preserve anonymous review.)