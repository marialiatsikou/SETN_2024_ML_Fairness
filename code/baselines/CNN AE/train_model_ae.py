import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
import random
import cnn_ae_models

seed_value= 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# GLOBEM
'''working_directory = 'GLOBEM/'
data_folder = '../SimCLR/GLOBEM'
input_shape = (28, 1390)  # (timesteps, channels)
'''

# MIMIC
'''working_directory = 'MIMIC/'
data_folder = '../SimCLR/MIMIC'  # data to fine-tune
input_shape = (48, 76)
'''


# MESA
working_directory = 'MESA/'
data_folder = '../SimCLR/MESA'  # data to fine-tune
input_shape = (101, 5)


dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)
if not os.path.exists(os.path.join(working_directory, 'img')):
    os.mkdir(os.path.join(working_directory, 'img'))


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


# training parameters
batch_size = 1024
decay_steps = 1000
monitor="val_loss"
mode = "min"

from keras.callbacks import Callback
class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")


def cnn_ae_train_model(ae_model_save_path, model, x_train, optimizer, batch_size, loss_callback, verbose=2):
    """
    Train a CNN Autoencoder algorithm

    Parameters:
        model
            the CNN Autoencoder model for feature learning 

        x_train
            the numpy array for training (no labels)
        
        optimizer
            the optimizer for training

        batch_size
            the batch size for mini-batch training

    Saves to h5 file the trained AE model
    """


    model.compile(optimizer=optimizer, loss="mse")
    if working_directory == 'MESA/':
        model.compile(optimizer=optimizer, loss="mean_absolute_error")
    model.summary()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(ae_model_save_path,
                                                         monitor=monitor, mode=mode, save_best_only=True,
                                                         save_weights_only=False, verbose=verbose
                                                         )
    history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[best_model_callback, es_callback, loss_callback], verbose=2)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    

tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
x_train = np_train[0]

ae_name = "cnn_ae.hdf5"

ae_model_save_path = os.path.join(working_directory, ae_name)

if working_directory == 'MESA/':
    model = cnn_ae_models.create_ae_model_mesa(input_shape, model_name="base_model")
else:
    model = cnn_ae_models.create_ae_model(input_shape, model_name="base_model")

loss_callback = LossCallback()
cnn_ae_train_model(ae_model_save_path, model, x_train, optimizer, batch_size, loss_callback, verbose=2)

