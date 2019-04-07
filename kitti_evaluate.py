'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import sys
import json
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
from datetime import datetime
from process_kitti import get_files
from collections import defaultdict

def get_test_splits(subdir):
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
    return im_list, source_list

def compare_results(plot_save_dir, X_test, X_hat, nt):
# Plot some predictions
    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
    plt.figure(figsize = (nt, 2*aspect_ratio))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
    for i in plot_idx:
        for t in range(nt):
            plt.subplot(gs[t])
            plt.imshow(X_test[i,t], cmap='gray', interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Actual', fontsize=10)
        
            plt.subplot(gs[t + nt])
            plt.imshow(X_hat[i,t], cmap='gray', interpolation='none')
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Predicted', fontsize=10)
        plt.savefig(os.path.join(plot_save_dir, 'plot_' + str(i) + '.png'))
        plt.clf()

def make_error_plot(err_dict, err_save_dir):
        for k, v in err_dict.items():
            plt.figure(figsize=(10,10))
            x = range(len(v))
            plt.step(x, v, color='r', alpha=0.2, where='post', label=k)
#            plt.fill_between(x, v, alpha=0.2, color='b')
            plt.xlabel('Timestep for '+k)
            plt.ylabel('MSE Error '+k)
#            plt.ylim([0.0, 1.05])
#            plt.xlim([0.0, 1.0])
            plt.title("MSE Error plot")
            plt.savefig(os.path.join(err_save_dir, k+'.png'), bbox_inches='tight')
            plt.clf()
            
n_plot = 40
batch_size = 10
nt = 10

subdir = 'total'

weights_file = os.path.join(WEIGHTS_DIR, subdir, 'prednet_ucsd_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, subdir, 'prednet_ucsd_model.json')
test_file = os.path.join(DATA_DIR, subdir, 'X_Test.hkl')
test_sources = os.path.join(DATA_DIR, subdir, 'sources_Test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

X_hat = np.squeeze(X_hat, axis=-1)
X_test = np.squeeze(X_test, axis=-1)
#
Xhat_filename = 'Xhat.npy'
Xtest_filename = 'Xtest.npy'
mse_videos_filename = 'mse_videos.json'
mse_frame_filename = 'mse_frame.json'
mse_prev_frame_filename = 'mse_prev_frame.json'
overall_mse_filename = 'predictions.txt'

pred_save_dir = 'prediction_plots'
err_save_dir = 'error_plots'
err_prev_save_dir = 'prev_frame_plots'

now = datetime.now
folder_now = now().strftime("%Y_%m_%d-%H%M")

if not os.path.exists(RESULTS_SAVE_DIR): 
    os.mkdir(RESULTS_SAVE_DIR)

if not os.path.exists(os.path.join(RESULTS_SAVE_DIR, subdir)):
    os.mkdir(os.path.join(RESULTS_SAVE_DIR, subdir))

if not os.path.exists(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now)):
    os.mkdir(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now))

if not os.path.exists(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, pred_save_dir)):
    os.mkdir(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, pred_save_dir))

if not os.path.exists(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_save_dir)):
    os.mkdir(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_save_dir))

if not os.path.exists(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_prev_save_dir)):
    os.mkdir(os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_prev_save_dir))
    
Xhat_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, Xhat_filename)
Xtest_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, Xtest_filename)
mse_videos_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, mse_videos_filename)
mse_frame_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, mse_frame_filename)
mse_prev_frame_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, mse_prev_frame_filename)
overall_mse_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, overall_mse_filename)
pred_save_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, pred_save_dir)
err_save_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_save_dir)
err_prev_save_path = os.path.join(RESULTS_SAVE_DIR, subdir, folder_now, err_prev_save_dir)

#
#X_test = np.load(r'ucsd_results\UCSDped1\2019_04_03-2040\Xtest.npy')
#X_hat = np.load(r'ucsd_results\UCSDped1\2019_04_03-2040\Xhat.npy')
#plt.imshow(X_test[11,6], cmap='gray', interpolation='none')
if subdir == 'total':
    im_list, source_list = get_test_splits(['UCSDped1', 'UCSDped2'])
else:
    im_list, source_list = get_test_splits([subdir])
curr_location = 0
possible_starts = defaultdict(list)
while curr_location < len(im_list) - nt + 1:
    if source_list[curr_location] == source_list[curr_location + nt - 1]:
        possible_starts[source_list[curr_location]].append(curr_location)
        curr_location += nt
    else:
        curr_location += 1
#
mse_videos = dict()
mse_model_frame = defaultdict(list)
mse_prev_frame = defaultdict(list)
i = 0
for k,v in sorted(possible_starts.items()):
    n_mini_clips = len(v)
    mse_model_video = np.mean( (X_test[i:i+n_mini_clips, 1:] - X_hat[i:i+n_mini_clips, 1:])**2 ).item()
    mse_prev_video = np.mean( (X_test[i:i+n_mini_clips, :-1] - X_test[i:i+n_mini_clips, 1:])**2 ).item()
    for j in range(n_mini_clips):
        for z in range(1,nt):
            mse_model_frame[k].append(np.mean( (X_test[i+j, z, :] - X_hat[i+j, z, :])**2 ).item())
            mse_prev_frame[k].append(np.mean( (X_test[i+j, z-1, :] - X_test[i+j, z, :])**2 ).item())
    mse_videos[k] = (mse_model_video, mse_prev_video)
    i += n_mini_clips

mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
    
with open(mse_videos_path, 'w') as fp:
    json.dump(mse_videos, fp, sort_keys=True, indent=4)

with open(mse_frame_path, 'w') as fp:
    json.dump(mse_model_frame, fp, sort_keys=True, indent=4)

with open(mse_prev_frame_path, 'w') as fp:
    json.dump(mse_prev_frame, fp, sort_keys=True, indent=4)
    
#np.save(Xhat_path, X_hat)
#np.save(Xtest_path, X_test)

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
f = open(overall_mse_path, 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("\n Previous Frame MSE: %f" % mse_prev)
f.close()
compare_results(pred_save_path, X_test, X_hat, nt)
make_error_plot(mse_model_frame, err_save_path)
#with open(r'C:\Users\karth\Documents\GitHub\prednet\ucsd_results\res\mse_prev_frame.json') as f:
#    mse_prev_frame = json.load(f)
make_error_plot(mse_prev_frame, err_prev_save_path)
