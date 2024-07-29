import csv
import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import random



seed_value= 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')

# Library scripts
import supervised_models
import supervised_utilities

# Dataset-specific GLOBEM
'''working_directory = 'GLOBEM/'
data_folder = '../SimCLR/GLOBEM'
input_shape = (28, 1390)  # (timesteps, channels)
output_shape = 2
monitor = 'val_loss'
'''

# Dataset-specific MIMIC
'''working_directory = 'MIMIC/'
data_folder = '../SimCLR/MIMIC'
input_shape = (48, 76)  # (timesteps, channels)
output_shape = 2
monitor = 'val_loss'
'''

# Dataset-specific MESA
working_directory = 'MESA/'
data_folder = '../SimCLR/MESA'
input_shape = (101, 5)  # (timesteps, channels)
output_shape = 2
monitor = 'val_loss'


dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)
if not os.path.exists(os.path.join(working_directory, 'img')):
    os.mkdir(os.path.join(working_directory, 'img'))

# Load preprocessed data

np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
            np.load(os.path.join(data_folder, 'train_y.npy')))

np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))

np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))

if working_directory == 'MESA/':
    np_train = (np.transpose(np_train[0], (0, 2, 1)), np_train[1])
    np_val = (np.transpose(np_val[0], (0, 2, 1)), np_val[1])
    np_test = (np.transpose(np_test[0], (0, 2, 1)), np_test[1])

print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

batch_size = 64
# decay_steps = 1000
added_layers = 2
epochs = 200
early_stopping = True
weighted = True

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

# Supervised Baseline Model
base_model = supervised_models.run_supervised_base_model(input_shape, output_shape, dense_layers=added_layers, model_name="supervised_baseline_model_{}".format(added_layers))

print(base_model.summary())

print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

# LOGGING
logdir = os.path.join("../../experiments_logs", "{}_{}_{}_supervised_e{}_l{}_es{}_bs{}_w{}"
                      .format(working_directory.replace('/', ''), start_time_str, epochs, monitor, added_layers, early_stopping, batch_size, weighted))
if not os.path.exists(logdir):
    os.makedirs(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

subfolder = "l{}_e{}_es{}_bs{}_w{}".format(added_layers, epochs, early_stopping, batch_size, weighted)
linear_eval_best_model_file_name = os.path.join(working_directory, subfolder, "supervised.finetuned.hdf5")

if not os.path.exists(os.path.join(working_directory, subfolder)):
    os.makedirs(os.path.join(working_directory, subfolder))

# linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_finetuned_l{added_layers}_hs{hidden_size}_e{total_epochs}_es{early_stopping}_bs{batch_size}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
                                                         monitor=monitor, mode='min', save_best_only=True,
                                                         save_weights_only=False, verbose=1,
                                                         )

if weighted:
    class_weight = supervised_utilities.get_class_weigths(np_train)


# Early-stopping to avoid overfitting
if early_stopping:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10)
    if weighted:
        training_history = base_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = base_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
        )
else:
    if weighted:
        training_history = base_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = base_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
        )

training_loss = training_history.history['loss']
validation_loss = training_history.history['val_loss']
epochs = range(1, len(training_loss) + 1)
# Plot the training and validation loss
plt.plot(epochs, training_loss, 'b', label='Training Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# best model
if working_directory != 'GLOBEM/':
    subfolder = 'l2_e100_esFalse_bs128_wTrue'
else:
    subfolder = 'l2_e200_esTrue_bs64_wTrue'

linear_eval_best_model_file_name = os.path.join(working_directory, subfolder, "supervised.finetuned.hdf5")
print("Loading file from: {}".format(linear_eval_best_model_file_name))
linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)

metrics = supervised_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
# Write metrics to file
with open(linear_eval_best_model_file_name.replace('.hdf5', '.csv'), 'w') as f:
    w = csv.writer(f)
    metrics.pop("Confusion Matrix")
    w.writerow(metrics.keys())
    w.writerow(metrics.values())

print("Results for test set from Model with the highest validation AUC:")
print(supervised_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Results for val set from Model with the highest validation AUC:")
print(supervised_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_val[0]), np_val[1], return_dict=True))




