import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import simclr_models
import simclr_utitlities


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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# SIMCLR finetuning
total_epochs = 100
# batch_size = 16
# added_layers = 2
# hidden_size = 128
early_stopping = True
#early_stopping = False
weighted = True
batch_size = 128  
monitor = 'val_loss'
mode = 'min'

# MIMIC 
# dataset = MIMIC
# working_directory = 'MIMIC/'
# data_folder = 'MIMIC'  # data to fine-tune
# pretrained_model_name = f'{seed}_simclr_200_scale_negate.hdf5'
# model_path = os.path.join(working_directory, pretrained_model_name)
# input_shape = (48, 76)
# output_shape = 2  # edit this to be the number of label classes

# working_directory = 'MESA/'
# data_folder = 'MESA'  # data to fine-tune
# input_shape = (101, 5)
# output_shape = 2  # edit this to be the number of label classes


# GLOBEM
dataset = 'GLOBEM'
working_directory = 'GLOBEM/'
data_folder = 'GLOBEM'
input_shape = (28, 1390)  #(timesteps, channels)
output_shape = 2  


# MIMIC
'''dataset = 'MIMIC'
working_directory = 'MIMIC/'
data_folder = 'MIMIC'
input_shape = (48, 76)  #(timesteps, channels)
output_shape = 2  
'''

# MESA
'''dataset = 'MESA'
working_directory = 'MESA/'
data_folder = 'MESA'
input_shape = (101, 5)  #(timesteps, channels)
output_shape = 2
'''

if dataset == 'GLOBEM':
    pretrained_model_name = f'{seed}_simclr_100_scale_negate.hdf5'
    model_path = os.path.join(working_directory, pretrained_model_name)


# Load preprocessed data
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
            np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))

if dataset == 'MESA':
    np_train = (np.transpose(np_train[0], (0, 2, 1)), np_train[1])
    np_val = (np.transpose(np_val[0], (0, 2, 1)), np_val[1])
    np_test = (np.transpose(np_test[0], (0, 2, 1)), np_test[1])

def train_model_freezing_alternatives(frozen_layers, first_last=False, middle=False):
    
    print("Current working directory: {}".format(os.getcwd()))

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    tf.keras.backend.set_floatx('float32')

    if first_last:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}_fl"
                          .format(dataset,start_time_str, total_epochs, early_stopping, batch_size, weighted, frozen_layers))
    elif middle:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}_m"
                              .format(dataset,start_time_str, total_epochs, early_stopping,
                                      batch_size, weighted, frozen_layers))
    else:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}"
                              .format(dataset,start_time_str, total_epochs, early_stopping,
                                      batch_size, weighted, frozen_layers))

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(simclr_model, output_shape,
                                                                                           intermediate_layer=7,
                                                                                           first_last=first_last,
                                                                                           middle=middle)
    print(full_evaluation_model.summary())
    if first_last:
        subfolder = "e{}_es{}_bs{}_w{}_f{}_fl".format(total_epochs,
                                                  early_stopping, batch_size, weighted, frozen_layers)
    elif middle:
        subfolder = "e{}_es{}_bs{}_w{}_f{}_m".format(total_epochs,
                                                         early_stopping, batch_size, weighted, frozen_layers)
    else:
        subfolder = "e{}_es{}_bs{}_w{}_f{}".format(total_epochs,
                                                         early_stopping, batch_size, weighted, frozen_layers)

    full_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.frozen.hdf5")
    if not os.path.exists(os.path.join(data_folder, subfolder)):
        os.makedirs(os.path.join(data_folder, subfolder))

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=full_eval_best_model_file_name,
                                                             monitor=monitor, mode=mode, save_best_only=True,
                                                             save_weights_only=False, verbose=1
                                                             )


    if weighted:
        class_weight = simclr_utitlities.get_class_weigths(np_train)
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5)
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback, es_callback],
                validation_data=np_val,
                class_weight=class_weight
            )
        else:
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback],
                validation_data=np_val,
                class_weight=class_weight
            )
    else:
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5)
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback, es_callback],
                validation_data=np_val,
            )
        else:
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback],
                validation_data=np_val,
            )

    full_evaluation_model.evaluate(
        x=np_test[0],
        y=np_test[1],
        return_dict=True
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

    
# re-read original base model
simclr_model = tf.keras.models.load_model(model_path)
frozen_layers=1
train_model_freezing_alternatives(frozen_layers=frozen_layers, middle=True)



# best model
subfolder = 'e100_esFalse_bs128_wTrue_f1_m'
full_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.frozen.hdf5")
full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name)
print("\"Supervised\" Model\n{}".format(full_eval_best_model.summary()))


print("Model with min val loss in val set:")
print(
    simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_val[0]), np_val[1], return_dict=True))

metrics = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
# Write metrics to file
with open(full_eval_best_model_file_name.replace('.hdf5', '.csv'), 'w') as f:
    w = csv.writer(f)
    metrics.pop("Confusion Matrix")
    w.writerow(metrics.keys())
    w.writerow(metrics.values())

print("Model with min val loss in test set:")
print(metrics)