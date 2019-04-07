'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
#import requests
#from bs4 import BeautifulSoup
#import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from kitti_settings import *
import os

desired_im_sz = (128, 160)


if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def get_files(folder_name, subdir):
    files = []
    for sd in subdir:
        path = os.path.join(DATA_DIR, sd, folder_name)
        files += [os.path.join(sd, folder_name, i) for i in os.listdir(path) if not i.startswith('.') and os.path.isdir(os.path.join(path,i))]
    return files

def process_data(subdir):
    splits = {s: get_files(s, subdir) for s in ['Train', 'Test', 'Val']} #['Test']
    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, folder) 
            if os.path.exists(im_dir):
                files = [i for i in os.listdir(im_dir) if not i.startswith('.')]
                im_list += [os.path.join(im_dir ,f) for f in sorted(files)]
                if len(subdir) == 2:
                    source_list += [os.path.basename(folder)+'_'+os.path.dirname(os.path.dirname(folder))] * len(files)
                else:
                    source_list += [os.path.basename(folder)] * len(files)
        im_list.sort()
        source_list.sort()
        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (1,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)
            
        if (len(subdir) == 1):        
            hkl.dump(X, os.path.join(DATA_DIR, subdir[0], 'X_' + split + '.hkl'))
            hkl.dump(source_list, os.path.join(DATA_DIR, subdir[0], 'sources_' + split + '.hkl'))
        elif (len(subdir) == 2):
            if not os.path.exists(os.path.join(DATA_DIR, 'total')):
                os.mkdir(os.path.join(DATA_DIR, 'total'))
            hkl.dump(X, os.path.join(DATA_DIR, 'total', 'X_' + split + '.hkl'))
            hkl.dump(source_list, os.path.join(DATA_DIR, 'total', 'sources_' + split + '.hkl')) 
            
#        X_saved = hkl.load(os.path.join(DATA_DIR, subdir[0], 'X_test.hkl'))
#        assert np.all((X, X_saved))

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im.reshape(desired_sz + (1,))


if __name__ == '__main__':
#    process_data(['UCSDped1'])
#    process_data(['UCSDped2'])
    process_data(['UCSDped1', 'UCSDped2'])
