import os
import pickle
import re
import numpy as np

import pandas as pd
import scipy
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from fairness_utilities import *
#from definitions import ROOT_DIR

from sklearn import metrics

import seaborn as sns
sns.set(font_scale = 1.5)

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

import simclr_models
import simclr_utitlities
import statistics


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set(style='white', context="poster", font='Work Sans Regural')

data_folder = 'MIMIC'
tag = 'supervised'
tag = 'cnn'
tag = 'simclr'
tag = 'cpc'


if tag=='supervised':
    working_directory = '../Supervised/MIMIC'
    subfolder = 'l2_e100_esFalse_bs128_wTrue'  
    model_name = 'supervised.finetuned.hdf5'

if tag=='cnn':
    working_directory = '../CNN AE/MIMIC'
    subfolder ='hs128_e200_esTrue_bs128_wTrue_rFalse'
    model_name = 'cnn_ae.finetuned.hdf5'
    #model_name = 'cnn_ae.frozen.hdf5'

if tag=='simclr':
    working_directory = '../SimCLR/MIMIC'
    subfolder ='e100_esFalse_bs128_wTrue_f1_m'
    model_name = 'simclr.frozen.hdf5'

if tag=='cpc':
    working_directory = '../CPC/MIMIC'

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

pd.Series(np_test[1][:,1]).value_counts()

if tag!='cpc':
    image_folder = os.path.join(working_directory, 'img', subfolder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
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
subjects = pd.read_csv(os.path.join('datasets', data_folder, 'demographics_rich.csv'))
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


# extract subject_id from stay
regex = r"(?:^\d+)"
df.loc[:, "SUBJECT_ID"] = df.stay.apply(lambda stay: re.search(regex, stay).group(0))
df.SUBJECT_ID = df.SUBJECT_ID.astype(int)
df.drop(['stay'], axis=1, inplace=True)
print(df.head(100))

len(pd.unique(df['SUBJECT_ID']))
df.groupby('SUBJECT_ID').count().max()

# join with demographics
df = df.merge(subjects, how='left', on='SUBJECT_ID')
print("Total rows after merging: {}".format(df.shape[0]))
category_counts = df['RELIGION'].value_counts()
print(category_counts)



# Calculate frequences
protected_attributes = ['LANGUAGE', 'INSURANCE', 'RELIGION', 'ETHNICITY', 'GENDER', 'AGE']
privileged_classes = [['ENGL'], ['Private'], ['CATHOLIC'], ['WHITE'], ['M'], [1]]



test_listfile.head(20)
# Get subject from test
test_listfile.loc[:, "subject"] = test_listfile.stay.apply(lambda stay: stay.split("_")[0]).astype(int)
test_listfile.head()

# merge demographics in test df
test_listfile = test_listfile.merge(subjects, left_on="subject", right_on="SUBJECT_ID", how="left")
test_listfile.head()
print("Test rows with null demograpphics: {} ({}%)".format(test_listfile.SUBJECT_ID.isna().sum(), test_listfile.SUBJECT_ID.isna().sum()/test_listfile.shape[0]))
test_listfile.drop(columns=['SUBJECT_ID'], inplace=True)
test_listfile['subject'] = test_listfile['subject'].astype(int)
test_listfile.head()
#add predictions
test_listfile.loc[:, "y_pred"] = predictions
test_listfile.head()
print(len(test_listfile))


subject_counts = test_listfile['subject'].value_counts()
print(subject_counts.head(50))
counts_list = subject_counts.values.tolist()
print(counts_list[:10])
count_of_ones = counts_list.count(1)
print("Number of counts with value 1:", count_of_ones)

import matplotlib.pyplot as plt

top_subject_counts = subject_counts.head(50)  
# Plot bar plot
plt.figure(figsize=(10, 6))
top_subject_counts.plot(kind='bar')
plt.title('Counts of Top Subjects')
plt.xlabel('Subject')
plt.ylabel('Count')
plt.xticks(rotation=90)  
plt.tight_layout()
plt.show()


# Plot histogram of subject counts
plt.figure(figsize=(10, 6))
plt.hist(subject_counts, bins=30)
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# Bias assessment for the entire dataset
# Overall performance
if tag!='cpc':
    train_predicted_labels = pd.Series(np.argmax(pretrained_model.predict(np_train[0]), axis=1))
    val_predicted_labels = pd.Series(np.argmax(pretrained_model.predict(np_val[0]), axis=1))
    test_predicted_labels = pd.Series(np.argmax(pretrained_model.predict(np_test[0]), axis=1))

    df_predicted_labels = pd.concat([train_predicted_labels, val_predicted_labels, test_predicted_labels])
    df_predicted_labels.reset_index(drop=True, inplace=True)
    df.loc[:, 'y_pred'] = df_predicted_labels
    df.head()
    for protected_attribute in protected_attributes:
        print("\n------------- ATTRIBUTE: {} -------------\n".format(protected_attribute))
        true_values, predictions = prepare_dataset_aif360(protected_attribute, df)
        metric_pred, classified_metric = aif360_model(true_values, predictions, protected_attribute, privileged_classes[protected_attributes.index(protected_attribute)], favorable_class=1)
        print_aif360_result(metric_pred, classified_metric)
        # evaluating in terms of accuracy
        print_aif360_accuracy_metrics(classified_metric)


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


final_metrics = ['average_absolute_odds_difference','equal_opportunity_difference', 'false_negative_rate_ratio', 'false_omission_rate_ratio', 'error_rate_ratio']

# Filter the rows based on the specified fairness metrics
df_fairness_metrics_final = df_fairness_metrics[df_fairness_metrics['fairness_metric'].isin(final_metrics)].copy()

if tag!='cpc':
    df_fairness_metrics_final.to_csv(os.path.join(working_directory, subfolder, model_name.replace('.hdf5', '_fairness_metrics.csv')), index=False)
else:
    df_fairness_metrics_final.to_csv('../CPC/MIMIC/CPC_fairness_metrics.csv', index=False)




