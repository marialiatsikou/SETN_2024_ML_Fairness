import os
import pickle
import re

import pandas as pd
import scipy
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from fairness_utilities import *
# from definitions import ROOT_DIR

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


data_folder = 'MESA'
tag = 'supervised'
#tag = 'cnn'
#tag = 'simclr'
#tag = 'cpc'


if tag=='supervised':
    working_directory = '../Supervised/MESA'
    subfolder = 'l2_e100_esFalse_bs128_wTrue'  
    model_name = 'supervised.finetuned.hdf5'

if tag=='cnn':
    working_directory = '../CNN AE/MESA'
    subfolder ='hs128_e200_esTrue_bs1024_wTrue_rFalse'
    model_name = 'cnn_ae.finetuned.hdf5'

if tag=='simclr':
    working_directory = '../SimCLR/MESA'
    subfolder ='e100_esFalse_bs128_wTrue_f1_m'
    model_name = 'simclr.frozen.hdf5'

if tag=='cpc':
    working_directory = '../CPC/MESA'

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


if tag!='cpc':
    #Load model    
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
    predictions, probs = get_metrics(model, data_loaders["test"], args)
    predictions = np.array(predictions)
    


for _ in range(3):
    os.chdir('..')
subjects = pd.read_csv(os.path.join('datasets', data_folder, 'demographics.csv'), delimiter=';')
# subject IDs per train-validation-test set
train_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'train_listfile.csv'))
val_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'val_listfile.csv'))
test_listfile = pd.read_csv(os.path.join('datasets', data_folder, 'test_listfile.csv'))
# change back the working directory
os.chdir(os.path.join('code', 'baselines', 'SimCLR'))
subjects.head()

df = pd.concat([train_listfile, val_listfile, test_listfile])
print("Total rows: {}".format(df.shape[0]))
print(df.head(100))
len(pd.unique(df.mesaid))

# join with demographics
print("Total rows before merging: {}".format(df.shape[0]))
df = df.merge(subjects, how='left', on='mesaid')
print("Total rows after merging: {}".format(df.shape[0]))

# Calculate frequences
df.loc[:, 'nsrr_age_gt65'] = df.nsrr_age.map(lambda age: 'no' if age < 65 else 'yes')
df.loc[:, 'AGE'] = df.nsrr_age.map(lambda age: '<65' if age < 65 else '>=65')
df.head()


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


#bias analysis
sns.set(style='white', font='Work Sans Regural')
# merge demographics in test df
test_listfile = test_listfile.merge(subjects, on="mesaid", how="left")
test_listfile.head()
test_listfile.loc[:, 'nsrr_age_gt65'] = test_listfile.nsrr_age.map(lambda age: 'no' if age < 65 else 'yes')
test_listfile.head()
print("Test rows with null demograpphics: {} ({}%)".format(test_listfile.mesaid.isna().sum(), test_listfile.mesaid.isna().sum()/test_listfile.shape[0]))
test_listfile.drop(columns=['mesaid'], inplace=True)
test_listfile.head()
test_listfile.rename({'wake':'y_true'}, axis=1, inplace=True)
test_listfile = test_listfile[['y_true', 'nsrr_age_gt89', 'nsrr_age_gt65', 'nsrr_sex', 'nsrr_race']]
test_listfile.loc[:, "y_pred"] = predictions



# Bias assessment for the entire dataset
# Overall performance
df.rename({'wake':'y_true'}, axis=1, inplace=True)
df = df[['mesaid', 'y_true', 'nsrr_age_gt89', 'nsrr_age_gt65', 'nsrr_sex', 'nsrr_race']]
df.head()

protected_attributes = ['nsrr_age_gt89', 'nsrr_age_gt65', 'nsrr_sex', 'nsrr_race']
privileged_classes = [['no'], ['no'], ['male'], ['white']]

# get predictions for df and test set
# Overall performance
if tag!='cpc':
    train_predicted_labels = pd.Series(np.argmax(train_predictions, axis=1))
    val_predicted_labels = pd.Series(np.argmax(validation_predictions, axis=1))
    test_predicted_labels = pd.Series(np.argmax(test_predictions, axis=1))
    df_predicted_labels = pd.concat([train_predicted_labels, val_predicted_labels, test_predicted_labels])
    df_predicted_labels.reset_index(drop=True, inplace=True)
    df.loc[:, 'y_pred'] = df_predicted_labels
    df.head()

# Bias assessment for test set

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

final_metrics = ['average_absolute_odds_difference', 'false_positive_rate_ratio',  'error_rate_ratio', 'average_predictive_value_difference']

# Filter the rows based on the specified fairness metrics
df_fairness_metrics_final = df_fairness_metrics[df_fairness_metrics['fairness_metric'].isin(final_metrics)].copy()

if tag!='cpc':
    df_fairness_metrics_final.to_csv(os.path.join(working_directory, subfolder, model_name.replace('.hdf5', '_fairness_metrics.csv')), index=False)
else:
    df_fairness_metrics_final.to_csv('../CPC/MESA/CPC_fairness_metrics.csv', index=False)
