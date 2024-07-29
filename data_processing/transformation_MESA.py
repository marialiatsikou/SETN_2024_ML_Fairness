import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import os

timesteps = 100
channels = 1

demographics = pd.read_csv("../datasets/MESA/demographics.csv", delimiter=";")
demographics = demographics[['mesaid', 'nsrr_age', 'nsrr_sex', 'nsrr_race']]
demographics.drop_duplicates(subset=['mesaid'], inplace=True)
demographics.head()

train = pd.read_csv("../datasets/MESA/dftrain_task1.csv")
test = pd.read_csv("../datasets/MESA/dftest_task1.csv")

scaler = StandardScaler()
scaler.fit(train[["activity"]].fillna(0.0))

train["activity"] = scaler.transform(train[["activity"]].fillna(0.0))
test["activity"] = scaler.transform(test[["activity"]].fillna(0.0))

# split into train and validation
def split_data(df, percent=0.2):
    uids = df.mesaid.unique()
    np.random.seed(42)
    np.random.shuffle(uids)
    test_position = int(uids.shape[0] * percent)

    uids_test, uids_train = uids[:test_position], uids[test_position:]

    # Splits dataset into training and test sets.
    # train_idx = wholedf[wholedf["mesaid"].apply(lambda x: x in uids_train)].index
    # dftrain = wholedf.iloc[train_idx].copy()
    dftrain = df[df['mesaid'].isin(uids_train)]

    # test_idx = wholedf[wholedf["mesaid"].apply(lambda x: x in uids_test)].index
    # dftest = wholedf.iloc[test_idx].copy()
    dftest = df[df['mesaid'].isin(uids_test)]
    return dftrain, dftest
train, val = split_data(train)

merged_train = train.merge(demographics, how='left', on='mesaid')
train_listfile = merged_train[['mesaid', 'wake']]
merged_val = val.merge(demographics, how='left', on='mesaid')
val_listfile = merged_val[['mesaid', 'wake']]
merged_test = test.merge(demographics, how='left', on='mesaid')
test_listfile = merged_test[['mesaid', 'wake']]

train_listfile.to_csv("../datasets/MESA/train_listfile.csv", index=False)
val_listfile.to_csv("../datasets/MESA/val_listfile.csv", index=False)
test_listfile.to_csv("../datasets/MESA/test_listfile.csv", index=False)

from time import sleep
def extract_x_y(df, seq_len, mesaid, feature="activity"):
    df = df[df["mesaid"] == mesaid][[feature, "gt"]].copy()
    # print(df)

    range_upper = int(seq_len/2 + 1)
    for s in range(1, range_upper):
	    df["shift_%d" % (s)] = df[feature].shift(s)

    for s in range(1, range_upper):
	    df["shift_-%d" % (s)] = df[feature].shift(-s)

    y = df["gt"]
    y = np.array([[1] if v else [0] for v in y])
    del df["gt"]
    x = df.fillna(-1).values
    return x,y

def get_data(df, seq_len):
    mesaids = df.mesaid.unique()
    features = ["activity", "whitelight", "redlight", "greenlight", "bluelight"]
    # 1st feature: activity
    print("Feature: {}".format(features[0]))
    x_, y_ = extract_x_y(df, seq_len, mesaids[0], feature=features[0])
    for mid in tqdm(mesaids[1:]):
        x_tmp, y_tmp = extract_x_y(df, seq_len, mid, feature=features[0])
        x_ = np.concatenate((x_, x_tmp))
        y_ = np.concatenate((y_, y_tmp))
    x_channels = x_
    x_channels = np.expand_dims(x_channels, axis=2)

    # remaining features
    for feature in features[1:]:
        print("Feature: {}".format(feature))
        x_, y_ = extract_x_y(df, seq_len, mesaids[0])
        for mid in tqdm(mesaids[1:]):
            x_tmp, y_tmp = extract_x_y(df, seq_len, mid, feature=feature)
            x_ = np.concatenate((x_, x_tmp))
            y_ = np.concatenate((y_, y_tmp))
        x_ = np.expand_dims(x_, axis=2)
        x_channels = np.concatenate([x_channels, x_], -1)
    return x_channels, y_


print("\nWindowing training data...\n")
x_train, y_train = get_data(train, timesteps)
print("\nWindowing validation data...\n")
x_val, y_val = get_data(val, timesteps)
print("\nWindowing test data...\n")
x_test, y_test = get_data(test, timesteps)

# TRAIN
train_samples = torch.from_numpy(x_train)
train_samples = torch.permute(train_samples, (0, 2, 1))
train_labels = torch.from_numpy(np.asarray(y_train).squeeze())
train_tensor = {'samples': train_samples, 'labels': train_labels}

# VAL
val_samples = torch.from_numpy(x_val)
val_samples = torch.permute(val_samples, (0, 2, 1))
val_labels = torch.from_numpy(np.asarray(y_val).squeeze())
val_tensor = {'samples': val_samples, 'labels': val_labels}

# TEST
test_samples = torch.from_numpy(x_test)
test_samples = torch.permute(test_samples, (0, 2, 1))
test_labels = torch.from_numpy(np.asarray(y_test).squeeze())
test_tensor = {'samples': test_samples, 'labels': test_labels}

# SAVE AS .PT
path = "../datasets/MESA/"
torch.save(train_tensor, os.path.join(path, "train.pt"))
torch.save(val_tensor, os.path.join(path, "val.pt"))
torch.save(test_tensor, os.path.join(path, "test.pt"))