'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from keras import backend as K
#from keras.models import Model, model_from_json
#from keras.layers import Input, Dense, Flatten
#
#from prednet import PredNet
#from data_utils import SequenceGenerator
from kitti_settings import *


n_plot = 40
batch_size = 10
nt = 10

#weights_file = os.path.join(WEIGHTS_DIR, 'prednet_ucsd_weights.hdf5')
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_ucsd_model.json')
#test_file = os.path.join(DATA_DIR, 'UCSDped2', 'X_Test.hkl')
#test_sources = os.path.join(DATA_DIR, 'UCSDped2', 'sources_Test.hkl')
#
## Load trained model
#f = open(json_file, 'r')
#json_string = f.read()
#f.close()
#train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
#train_model.load_weights(weights_file)
#
## Create testing model (to output predictions)
#layer_config = train_model.layers[1].get_config()
#layer_config['output_mode'] = 'prediction'
#data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
#test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
#input_shape = list(train_model.layers[0].batch_input_shape[1:])
#input_shape[0] = nt
#inputs = Input(shape=tuple(input_shape))
#predictions = test_prednet(inputs)
#test_model = Model(inputs=inputs, outputs=predictions)
#
#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
#X_test = test_generator.create_all()
#X_hat = test_model.predict(X_test, batch_size)
#if data_format == 'channels_first':
#    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
#    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

from process_kitti import get_files
from collections import defaultdict

subdir = 'UCSDped1'

splits = {s: get_files(s, subdir) for s in ['Test']}    
for split in splits:
    im_list = []
    source_list = []  # corresponds to recording that image came from
    for folder in splits[split]:
        for sd in subdir:
            im_dir = os.path.join(DATA_DIR, sd, split, folder)
            if os.path.exists(im_dir):
                files = [i for i in os.listdir(im_dir) if not i.startswith('.')]
                im_list += [os.path.join(im_dir ,f) for f in sorted(files)]
                source_list += [folder] * len(files)

X_test = np.load(r'C:\Users\karth\Documents\GitHub\prednet\X_test_ped1.npy')
X_hat = np.load(r'C:\Users\karth\Documents\GitHub\prednet\X_hat_ped1.npy')

curr_location = 0
possible_starts = defaultdict(list)
while curr_location < X_test.shape[0] - nt + 1:
    if source_list[curr_location] == source_list[curr_location + nt - 1]:
        possible_starts[source_list[curr_location]].append(curr_location)
        curr_location += nt
    else:
        curr_location += 1

mse_videos = dict()
i = 0
for k,v in possible_starts.items():
    n_mini_clips = len(v)
    mse_model_video = np.mean( (X_test[i:i+n_mini_clips+1, 1:] - X_hat[i:i+n_mini_clips+1, 1:])**2 )
    mse_prev_video = np.mean( (X_test[i:i+n_mini_clips+1, :-1] - X_test[i:i+n_mini_clips+1, 1:])**2 )
    mse_videos[k] = (mse_model_video, mse_prev_video)
    i += n_mini_clips
## Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )

#if not os.path.exists(RESULTS_SAVE_DIR): 
#    os.mkdir(RESULTS_SAVE_DIR)
#f = open(RESULTS_SAVE_DIR + 'prediction_scores_ped2.txt', 'w')
#f.write("Model MSE: %f\n" % mse_model)
#f.write("Previous Frame MSE: %f" % mse_prev)
#f.close()
#
#X_test_file = os.path.join(RESULTS_SAVE_DIR, 'X_test_ped2')
#X_hat_file = os.path.join(RESULTS_SAVE_DIR, 'X_hat_ped2')
#X_hat = np.squeeze(X_hat, axis=-1)
#X_test = np.squeeze(X_test, axis=-1)
#
#np.save(X_test_file, X_test)
#np.save(X_hat_file, X_hat)
# Plot some predictions
#aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
#plt.figure(figsize = (nt, 2*aspect_ratio))
#gs = gridspec.GridSpec(2, nt)
#gs.update(wspace=0., hspace=0.)
#plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots_ped2/')
#if not os.path.exists(plot_save_dir): 
#    os.mkdir(plot_save_dir)
#plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
#for i in plot_idx:
#    for t in range(nt):
#        plt.subplot(gs[t])
#        plt.imshow(X_test[i,t], cmap='gray', interpolation='none')
#        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#        if t==0: plt.ylabel('Actual', fontsize=10)
#
#        plt.subplot(gs[t + nt])
#        plt.imshow(X_hat[i,t], cmap='gray', interpolation='none')
#        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#        if t==0: plt.ylabel('Predicted', fontsize=10)
#
#    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#    plt.clf()
