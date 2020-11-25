from __future__ import print_function

from model import unet
from data_generator import data_generator
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


def train():

    """
    A function to perform training by initializing the values of the model object.
    ModelCheckpoint is used to save the trained weights.
    fit_generator starts the training process.

    """
    model = unet(in_shape=(256, 512, 3), num_classes=7, lrate=1e-4, decay_rate=5e-4, vgg_path= None, dropout_rate=0.5)

    # Define callbacks
    mod_save = ModelCheckpoint(filepath='gpu_exp_6classes_model_weight.h5', save_weights_only=True)

    # training
    model.fit_generator(data_generator('data.h5', 4, 'train'),
                        steps_per_epoch=2975//4,
                        validation_data=data_generator('data.h5', 4, 'val'),
                        validation_steps=500/4,
                        callbacks=[mod_save], 
                        epochs=100,
                        verbose=1)

if __name__ == "__main__":
    train()
