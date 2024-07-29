import csv
import os
import pickle
import datetime
import numpy as np
import tensorflow as tf

import random

import cnn_ae_models
import cnn_ae_utilities

seed_value= 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')


# MIMIC
working_directory = 'MIMIC/'
data_folder = '../SimCLR/MIMIC'  # data to fine-tune
input_shape = (48, 76)
output_shape = 2  # edit this to be the number of label classesmonitor = "val_recall"
monitor = "val_loss"
mode = 'min'


'''working_directory = 'MESA/'
data_folder = '../SimCLR/MESA'  # data to fine-tune
input_shape = (101, 5)
output_shape = 2  # edit this to be the number of label classes
monitor = "val_loss"
mode = 'min'
'''

# GLOBEM
'''working_directory = 'GLOBEM/'
data_folder = '../SimCLR/GLOBEM'
input_shape = (28, 1390)
output_shape = 2  # edit this to be the number of label classes
monitor = "val_loss"
mode = 'min'
'''


dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

total_epochs = 200
batch_size = 128
hidden_size = 128

#layers of the encoder
number_of_encoder_layers = 3

early_stopping = True
weighted = True
resampling = False


# Load preprocessed data
if resampling:
    np_train = (np.load(os.path.join(data_folder, 'train_resampled_x.npy')),
                np.load(os.path.join(data_folder, 'train_resampled_y.npy')))
else:
    np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
                np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))
start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')


if working_directory == 'MESA/':
    np_train = (np.transpose(np_train[0], (0, 2, 1)), np_train[1])
    np_val = (np.transpose(np_val[0], (0, 2, 1)), np_val[1])
    np_test = (np.transpose(np_test[0], (0, 2, 1)), np_test[1])


ae_name = "cnn_ae.hdf5"
trained_ae = tf.keras.models.load_model(os.path.join(working_directory, ae_name))

# Get the encoder layers from the autoencoder
encoder_layers = trained_ae.layers[:number_of_encoder_layers]  
encoder_model = tf.keras.Sequential(encoder_layers)
classifier = cnn_ae_models.create_full_classification_model(input_shape, encoder_model, output_shape, hidden_size=hidden_size)
lr = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=1000)
classifier.compile(
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.03),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    
)
classifier.summary()

subfolder = "hs{}_e{}_es{}_bs{}_w{}_r{}".format(hidden_size, total_epochs, early_stopping, batch_size, weighted, resampling)
best_model_file_name = os.path.join(data_folder, subfolder, "cnn_ae.finetuned.hdf5")

if not os.path.exists(os.path.join(data_folder, subfolder)):
    os.makedirs(os.path.join(data_folder, subfolder))

best_model_callback = tf.keras.callbacks.ModelCheckpoint(best_model_file_name,
                                                         monitor=monitor, mode=mode, save_best_only=True,
                                                         save_weights_only=False, verbose=2,
                                                         )
if weighted:
    class_weight = cnn_ae_utilities.get_class_weigths(np_train)


# Early-stopping to avoid overfitting
if early_stopping:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5)
    if weighted:
        training_history = classifier.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, 
                       es_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = classifier.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback,
                       es_callback],
            validation_data=np_val,
        )
else:
    if weighted:
        training_history = classifier.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback,
                    ],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = classifier.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, 
                       ],
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
if working_directory == 'GLOBEM/':
    best_model_file_name = os.path.join(data_folder, "hs128_e200_esTrue_bs128_wTrue_rFalse/cnn_ae.finetuned.hdf5")
if working_directory == 'MIMIC/':
    best_model_file_name = os.path.join(data_folder, "hs128_e200_esTrue_bs128_wTrue_rFalse/cnn_ae.finetuned.hdf5")
    #best_model_file_name = os.path.join(data_folder, "hs128_e200_esTrue_bs128_wTrue_rFalse/cnn_ae.frozen.hdf5")
if working_directory == 'MESA/':
    best_model_file_name = os.path.join(data_folder, "hs128_e200_esTrue_bs1024_wTrue_rFalse/cnn_ae.finetuned.hdf5")


print("Loading file from: {}".format(best_model_file_name))

linear_eval_best_model = tf.keras.models.load_model(best_model_file_name)
metrics = cnn_ae_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
# Write metrics to file
with open(best_model_file_name.replace('.hdf5', '.csv'), 'w') as f:
    w = csv.writer(f)
    metrics.pop("Confusion Matrix")
    w.writerow(metrics.keys())
    w.writerow(metrics.values())

print("Model with min val loss in the test set:")
print(metrics)
#print("Model with min val loss in the validation set:")
#print(cnn_ae_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_val[0]), np_val[1], return_dict=True))

