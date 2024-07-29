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

import simclr_models
import simclr_utitlities

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set(style='white', context="poster", font='Work Sans Regural')



data_folder = 'GLOBEM'
tag = 'supervised'
#tag = 'cnn'
#tag = 'simclr'
#tag = 'cpc'


if tag=='supervised':
    working_directory = '../Supervised/GLOBEM'
    subfolder = 'l2_e200_esTrue_bs64_wTrue'  
    model_name = 'supervised.finetuned.hdf5'

if tag=='cnn':
    working_directory = '../CNN AE/GLOBEM'
    subfolder ='hs128_e200_esTrue_bs128_wTrue_rFalse'
    model_name = 'cnn_ae.finetuned.hdf5'

if tag=='simclr':
    working_directory = '../SimCLR/GLOBEM'
    subfolder ='e100_esFalse_bs128_wTrue_f1_m'
    model_name = 'simclr.frozen.hdf5'

if tag=='cpc':
    working_directory = '../CPC/GLOBEM'

# Load preprocessed data
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
        np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
        np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
        np.load(os.path.join(data_folder, 'test_y.npy')))
print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))


if tag!='cpc':

    pretrained_model = tf.keras.models.load_model(os.path.join(working_directory, subfolder, model_name), compile=False)  # compile=False as we use the model only for inference
    pretrained_model.summary()
    probs = pretrained_model.predict(np_test[0])
    predictions = np.argmax(probs, axis=1)

else:
    os.chdir('../CPC')
    from utils import load_classifier_model 
    from ubi_dataset import load_dataset
    from arguments import parse_args
    from evaluate_with_classifier import get_metrics

    args = parse_args()
    print(args)
    model = load_classifier_model(args=args)
    data_loaders, dataset_sizes = load_dataset(args, classifier=True)
    predictions,_ = get_metrics(model, data_loaders["test"], args)
    predictions = np.array(predictions)
    

for _ in range(3):
    os.chdir('..')
subjects = pd.read_csv(os.path.join('datasets', data_folder, 'demographics.csv'), delimiter=',')
# subject IDs per train-validation-test set
train_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'train_listfile.csv'))
val_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'val_listfile.csv'))
test_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'test_listfile.csv'))
subjects.head()
# change back the working directory
os.chdir(os.path.join('code', 'baselines', 'SimCLR'))

train_listfile.shape

df = pd.concat([train_listfile, val_listfile, test_listfile])
print("Total rows: {}".format(df.shape[0]))
print(df.head(100))

df.PID.nunique()

df.gender = df.gender.apply(lambda v: 'Male' if v==1 else ('Female' if v==2 else ('Transgender' if v==3 else 'Genderqueer ' if v==4 else 'Other')))
df.disability = df.disability.apply(lambda v: 'Yes' if v==1 else 'No')

protected_attributes = ['gender', 'race', 'disability']
privileged_classes = [['Male'], ['White'], ['No']]

# Overall performance
if tag!='cpc':
    print("\n--- Train Performance Overall ---")
    train_predictions = pretrained_model.predict(np_train[0])
    train_performance = simclr_utitlities.evaluate_model_simple(train_predictions, np_train[1], return_dict=True)
    print(train_performance)
    print("\n--- Validation Performance Overall ---")
    validation_predictions = pretrained_model.predict(np_val[0])
    validation_performance = simclr_utitlities.evaluate_model_simple(validation_predictions, np_val[1], return_dict=True)
    print(validation_performance)
    print("\n--- Test Performance Overall ---")
    test_predictions = pretrained_model.predict(np_test[0])
    test_performance = simclr_utitlities.evaluate_model_simple(test_predictions, np_test[1], return_dict=True)
    print(test_performance)

#Bias analysis
sns.set(style='white', font='Work Sans Regural')
test_listfile

test_listfile.gender = test_listfile.gender.apply(lambda v: 'Male' if v==1 else ('Female' if v==2 else ('Transgender' if v==3 else 'Genderqueer ' if v==4 else 'Other')))
test_listfile.disability = test_listfile.disability.apply(lambda v: 'Yes' if v==1 else 'No')
test_listfile.drop(columns=['generation', 'age', 'year'], inplace=True)
test_listfile

# Plot histogram of subject counts
subject_counts = test_listfile['PID'].value_counts()
plt.figure(figsize=(10, 6))
plt.hist(subject_counts, bins=30)
plt.xlabel('Number of Entries')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig("results/globem_hist.png",bbox_inches='tight', dpi=600)

plt.show()


subject_counts_counter = Counter(test_listfile['PID'])
count_of_counts = {}

# Count how many unique IDs have a count of 1, 2, etc.
for count in subject_counts_counter.values():
    count_of_counts[count] = count_of_counts.get(count, 0) + 1
for count, number_of_ids in sorted(count_of_counts.items()):
    print(f"{number_of_ids} IDs have {count} entries")


test_listfile.loc[:, "y_pred"] = predictions


# Bias assessment for test set
test_listfile.loc[:, 'y_true'] = np_test[1][:,1]
test_listfile

df_protected = []
df_metric = []
df_value = []

for protected_attribute in protected_attributes:
    print("\n------------- ATTRIBUTE: {} -------------\n".format(protected_attribute))
    true_values, predictions = prepare_dataset_aif360(protected_attribute, test_listfile)
    metric_pred, classified_metric = aif360_model(true_values, predictions, protected_attribute, privileged_classes[protected_attributes.index(protected_attribute)], favorable_class=1)
    fairness_metrics = print_aif360_result(metric_pred, classified_metric)
    df_protected = df_protected + [protected_attribute for x in range(len(fairness_metrics))]
    df_metric = df_metric + list(fairness_metrics.keys())
    df_value = df_value + list(fairness_metrics.values())
    # evaluating in terms of accuracy
    print_aif360_accuracy_metrics(classified_metric)

df_fairness_metrics = pd.DataFrame({'protected_attribute': df_protected, 'fairness_metric': df_metric, 'value': df_value, 'tag': tag})

final_metrics = ['average_absolute_odds_difference', 'equal_opportunity_difference', 'false_negative_rate_ratio', 'false_omission_rate_ratio', 'error_rate_ratio']

# Filter the rows based on the specified fairness metrics
df_fairness_metrics_final = df_fairness_metrics[df_fairness_metrics['fairness_metric'].isin(final_metrics)].copy()

if tag!='cpc':
    df_fairness_metrics_final.to_csv(os.path.join(working_directory, subfolder, model_name.replace('.hdf5', '_fairness_metrics.csv')), index=False)
else:
    df_fairness_metrics_final.to_csv('../CPC/GLOBEM/CPC_fairness_metrics.csv', index=False)





