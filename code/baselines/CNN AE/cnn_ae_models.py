import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt


def create_ae_model(input_shape, model_name="cnn_ae_model"):

    model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv1D(
            filters=64, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1D(
            filters=32, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        
        layers.Conv1DTranspose(
            filters=32, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1DTranspose(
            filters=64, kernel_size=5, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=input_shape[1], kernel_size=5,  padding="same"),
    ]
)  
    return model


def create_ae_model_mesa(input_shape, model_name="cnn_ae_model"):

    model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv1D(
            filters=64, kernel_size=5, padding="valid", strides=2,
              activation="relu",  kernel_initializer='glorot_uniform'
        ),
        layers.Dropout(rate=0.1),
        #layers.BatchNormalization(),
        layers.Conv1D(
            filters=32, kernel_size=5, padding="valid", strides=2, 
            activation="relu", kernel_initializer='glorot_uniform'
        ),
        layers.Conv1DTranspose(
            filters=32, kernel_size=5, padding="valid", strides=2, 
            activation="relu", kernel_initializer='glorot_uniform'
        ),
        layers.Dropout(rate=0.1),
        #layers.BatchNormalization(),
        layers.Conv1DTranspose(
            filters=64, kernel_size=5, padding="valid", strides=2, 
            activation="relu", kernel_initializer='glorot_uniform'
        ),
        layers.Conv1DTranspose(filters=input_shape[1], kernel_size=5,  padding="same"),
    ]
)  
    return model


def create_full_classification_model(input_shape, encoder, output_shape, hidden_size=128):


    classification_model = keras.Sequential([
    layers.Input(shape=input_shape),
    encoder,
    layers.GlobalMaxPool1D(),  
    layers.Dropout(0.3),
    layers.Dense(hidden_size, activation='relu',
            kernel_regularizer=keras.regularizers.l2(l=1e-2)),
    layers.Dropout(0.3),
    layers.Dense(output_shape, activation='softmax') 
])
    
    return classification_model
