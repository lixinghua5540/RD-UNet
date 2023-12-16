from __future__ import print_function

import os


import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import cloud_net_model
from generators import mybatch_generator_train, mybatch_generator_validation
from losses import mloss
from utils import ADAMLearningRateTracker
from utils import get_input_image_names


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.compile(optimizer=Adam(lr=starting_learning_rate), loss=mloss, metrics=["accuracy"])

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(experiment_name + '_log_1.log')

    train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img, train_msk,
                                                                                      test_size=val_ratio,
                                                                                      random_state=42, shuffle=True)

    if train_resume:
        model.load_weights(weights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")


    model.fit_generator(
        generator=mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        validation_steps=np.ceil(len(val_img_split) / batch_sz),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate), csv_logger],
        # workers=8,
        # use_multiprocessing=True
    )



GLOBAL_PATH = 'path'
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'TrainingSPARCS')
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'TestSPARCS')

in_rows = 192
in_cols = 192
num_of_channels = 10
num_of_classes = 1
starting_learning_rate = 1e-4
end_learning_rate = 1e-8
max_num_epochs = 2000
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 10
max_bit = 65535
experiment_name = "RD-UNet"
weights_path = os.path.join(GLOBAL_PATH, experiment_name + '.h5')
train_resume = False

train_patches_csv_name = 'training_img.csv'
df_train_img = pd.read_csv(os.path.join(TRAIN_FOLDER, train_patches_csv_name))
train_img, train_msk = get_input_image_names(df_train_img, TRAIN_FOLDER, if_train=True)

train()

