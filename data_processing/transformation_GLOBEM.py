import numpy as np
import pandas as pd
import os
import torch
import sys

# get the path to the directory containing 'definitions.py'
definitions_dir = os.path.dirname(os.path.abspath('../definitions.py'))

# add the directory to sys.path
sys.path.append(definitions_dir)

from definitions import ROOT_DIR

os.chdir(ROOT_DIR)

train = pd.read_pickle('datasets/GLOBEM/train.pkl') #a tuple where samples are the 1st element & labels the 2nd
val = pd.read_pickle('datasets/GLOBEM/val.pkl')
test = pd.read_pickle('datasets/GLOBEM/test.pkl')

# TRAIN
train_samples = torch.from_numpy(train[0])
#train_labels= train[1][:,1] #for CPC only
train_labels = torch.from_numpy(train[1])
train_tensor = {'samples': train_samples, 'labels': train_labels}

# VAL
val_samples = torch.from_numpy(val[0])
#val_labels= val[1][:,1] #for CPC only
val_labels = torch.from_numpy(val[1])
val_tensor = {'samples': val_samples, 'labels': val_labels}

# TEST
test_samples = torch.from_numpy(test[0])
#test_labels= test[1][:,1]  #for CPC only
test_labels = torch.from_numpy(test[1])
test_tensor = {'samples': test_samples, 'labels': test_labels}


# SAVE AS .PT
path = "datasets/GLOBEM/"
torch.save(train_tensor, os.path.join(path, "train.pt"))
#torch.save(train_resampled_tensor, os.path.join(path, "train_resampled.pt"))
torch.save(val_tensor, os.path.join(path, "val.pt"))
torch.save(test_tensor, os.path.join(path, "test.pt"))

