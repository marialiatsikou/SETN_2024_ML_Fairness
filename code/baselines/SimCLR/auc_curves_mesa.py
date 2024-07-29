import os
import pickle
import re

import pandas as pd
import scipy
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter


from fairness_utilities import *

from sklearn import metrics

import seaborn as sns
sns.set(font_scale = 1.5)

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set(style='white', context="poster", font='Work Sans Regural')
original_directory = os.getcwd()

data_folder = 'MESA'
# Load preprocessed data
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
        np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
        np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
        np.load(os.path.join(data_folder, 'test_y.npy')))
np_train = (np.transpose(np_train[0], (0, 2, 1)), np_train[1])
np_val = (np.transpose(np_val[0], (0, 2, 1)), np_val[1])
np_test = (np.transpose(np_test[0], (0, 2, 1)), np_test[1])
print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

tags = ['1D CONV', '1D CONV AE', 'SimCLR', 'CPC']
probs_list = []


for tag in tags:
    if tag=='CPC':
        working_directory = '../CPC/MESA'
        os.chdir('../CPC')
        from utils import load_classifier_model 
        from ubi_dataset import load_dataset
        from arguments import parse_args
        from evaluate_with_classifier import get_metrics

        args = parse_args()
        print(args)
        model = load_classifier_model(args=args)
        data_loaders, dataset_sizes = load_dataset(args, classifier=True)
        predictions, probs = get_metrics(model, data_loaders["test"], args)
        probs_list.append(np.array(probs)) #for CPC the shape is (1083,)
        os.chdir(original_directory)

    else:
        if tag=='1D CONV':
            working_directory = '../Supervised/MESA'
            subfolder = 'l2_e100_esFalse_bs128_wTrue'  
            model_name = 'supervised.finetuned.hdf5'

        if tag=='1D CONV AE':
            working_directory = '../CNN AE/MESA'
            subfolder ='hs128_e200_esTrue_bs1024_wTrue_rFalse'
            model_name = 'cnn_ae.finetuned.hdf5'

        if tag=='SimCLR':
            working_directory = '../SimCLR/MESA'
            subfolder ='e100_esFalse_bs128_wTrue_f1_m'
            model_name = 'simclr.frozen.hdf5'
        
        pretrained_model = tf.keras.models.load_model(os.path.join(working_directory, subfolder, model_name), compile=False)  # compile=False as we use the model only for inference
        probs = pretrained_model.predict(np_test[0])
        probs_list.append(probs[:,1])
        #probs_list.append(predictions)


sns.set_context('poster')
for i in range(len(probs_list)):

    fpr, tpr, _ = metrics.roc_curve(
    np_test[1][:,1],
    probs_list[i])
    auc = np.round(metrics.roc_auc_score(np_test[1][:,1],probs_list[i], average='micro'), 3)
    
    plt.plot(fpr, tpr, label=tags[i]+", AUC={}".format(str(auc)))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(False)
    sns.despine()
    plt.legend(loc='lower right', fontsize=14)

plt.savefig("../results/auc_curve_mesa.png",bbox_inches='tight', dpi=300)



def merge_images_horiz(image1_path, image2_path, image3_path, output_path):
    '''merges barplots and boxplots'''

    image1 = plt.imread(image1_path)
    image2 = plt.imread(image2_path)
    image3 = plt.imread(image3_path)

    fig, axs = plt.subplots(1, 3, figsize=(5, 15))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[2].imshow(image3)

    # Add labels for each subplot
    axs[0].set_title("GLOBEM", fontsize=4)
    axs[1].set_title("MIMIC", fontsize=4)
    axs[2].set_title("MESA", fontsize=4)

    # Adjust layout to reduce empty space between subplots
    plt.subplots_adjust(wspace=0.02)

    plt.savefig(output_path, bbox_inches='tight',dpi=600)



image1_path = "../results/auc_curve_globem.png"
image2_path = "../results/auc_curve_mimic.png"
image3_path = "../results/auc_curve_mesa.png"
output_path = "../results/all_auc_curves.png"
merge_images_horiz(image1_path, image2_path, image3_path, output_path)
