import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import random

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

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# Library scripts

import simclr_models
import simclr_utitlities
import transformations

# Dataset-specific MIMIC
# working_directory = 'MIMIC/'
# data_folder = 'MIMIC'
# input_shape = (48, 76)  # (timesteps, channels)

# Dataset-specific MESA
# working_directory = 'MESA/'
# data_folder = 'MESA'
# input_shape = (101, 5)  # (timesteps, channels)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# GLOBEM
working_directory = 'GLOBEM/'
data_folder = 'GLOBEM'
input_shape = (28, 1390)  # (timesteps, channels)

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

print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

# SIMCLR training parameters
batch_size = 64
decay_steps = 1000
epochs = 100
temperature = 0.1
transform_funcs = [
    transformations.scaling_transform_vectorized,
    transformations.negate_transform_vectorized
]
transform_funcs_str = "scale_negate"
transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

# Base Model: A neural network base encoder, which is responsible for encoding the data samples into a latent space
base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
# SimCLR Head: A projection head, which is effectively another neural network that projects the representations in the
# latent space into another space for contrastive learning
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()

# LOGGING
logdir = "../experiments_logs/" + start_time_str
print("Started training...")
trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer,
                                                                          batch_size, transformation_function,
                                                                          temperature=temperature, epochs=epochs,
                                                                          is_trasnform_function_vectorized=True,
                                                                          verbose=1)

simclr_model_save_path = os.path.join(working_directory, f"{seed}_simclr_{epochs}_{transform_funcs_str}.hdf5")
print("Trained SimCLR model summary\n{}".format(trained_simclr_model.summary()))
trained_simclr_model.save(simclr_model_save_path)

# plotting loss
plt.figure(figsize=(12, 8))
plt.plot(epoch_losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(os.path.join(working_directory, "img", "epoch_loss_plot_{}.png".format(start_time_str)))
