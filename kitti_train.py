'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.layers import Dropout
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
from datetime import datetime
import json

subdir = 'UCSDped1' #'UCSDped1' 'total'
save_model = True  # if weights will be saved
if not os.path.exists(os.path.join(WEIGHTS_DIR, subdir)):
    os.mkdir(os.path.join(WEIGHTS_DIR, subdir))
    
weights_file = os.path.join(WEIGHTS_DIR, subdir, 'prednet_ucsd_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, subdir, 'prednet_ucsd_model.json')

# Data files
train_file = os.path.join(DATA_DIR, subdir, 'X_Train.hkl')
train_sources = os.path.join(DATA_DIR, subdir, 'sources_Train.hkl')
val_file = os.path.join(DATA_DIR, subdir, 'X_Val.hkl')
val_sources = os.path.join(DATA_DIR, subdir, 'sources_Val.hkl')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.exists(os.path.join(LOG_DIR, subdir)):
    os.mkdir(os.path.join(LOG_DIR, subdir))

now = datetime.now
folder_now = now().strftime("%Y_%m_%d-%H%M")

if not os.path.exists(os.path.join(LOG_DIR, subdir, folder_now)):
    os.mkdir(os.path.join(LOG_DIR, subdir, folder_now))

training_log = os.path.join(LOG_DIR, subdir, folder_now, 'log.csv')
model_weights = os.path.join(LOG_DIR, subdir, folder_now, 'weights.h5')
hyperparam = os.path.join(LOG_DIR, subdir, folder_now, 'hyperparam.json')
# Training parameters
nb_epoch = 50
batch_size = 4
samples_per_epoch = 600# 900, 600, 250
N_seq_val = 80#100, 80, 30  # number of sequences to use for validation
old_learning_rate = 0.001
new_learning_rate = 0.0007
epoch_learning_rate_number = 30

# Model parameters
n_channels, im_height, im_width = (1, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
sz1 = 32 #48
sz2 = 64 #96
sz3 = 128 #192
stack_sizes = (n_channels, sz1, sz2, sz3)
R_stack_sizes = stack_sizes
fz = 3
A_filt_sizes = (fz, fz, 1)
Ahat_filt_sizes = (fz, fz, fz, 1)
R_filt_sizes = (fz, fz, fz, 1)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10#5  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

hyperparam_dict = {'epoch': nb_epoch, 'batch_size': batch_size, 'samples_per_epoch': samples_per_epoch, 'N_seq_val': N_seq_val, 
                   'stack_sz': sz1, 'stack_sz2': sz2, 'stack_sz3': sz3, 'A_filt_sz': fz, 'old_learning_rate':old_learning_rate, 
                   'new_learning_rate':new_learning_rate, 'epoch_learning_rate':epoch_learning_rate_number}
with open(hyperparam, 'w') as f:
    json.dump(hyperparam_dict, f)

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors = Dropout(0.2)(errors)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: old_learning_rate if epoch < epoch_learning_rate_number else new_learning_rate    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): 
        os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))
    callbacks.append(CSVLogger(training_log))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)
model.save(model_weights)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
