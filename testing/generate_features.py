# In order to speed up the process, features for all windows in a frame that the SVC says might have a car,
# are saved to a folder ./project_video-frames/car_windows

# car_finder.py later on uses that to read features rather than finding them again

import matplotlib.pyplot as plt
import cv2, os, time, pickle, shutil, glob
import numpy as np
from lesson_functions import *
from car_finder import SingleFrameCarFinder

frame_folder = 'project_video-frames'
output_folder = "{}/car_windows/".format(frame_folder)
if os.path.exists(output_folder):
    print('removing folder', output_folder)
    shutil.rmtree(output_folder)

os.mkdir(output_folder)
print('Folder created', output_folder)

with open('svm-yuv-16x2x11.p', 'rb') as f:
    data = pickle.load(f)
    svc = data['svm']
    X_scaler = data['scaler']
    params = data['params']
    
print('Data Loaded')
print('SVC:', svc)
print('Params', params)

files = glob.glob('./{}/*.jpg'.format(frame_folder))
for index, filename in enumerate(files):
    # read RGB since thats what video will give us and then our function
    # internally converts it to LAB
    img_cs = load_image(filename, params['color_space'])

    cf = SingleFrameCarFinder(svc, X_scaler, params)

    t1 = time.time()
    car_windows, window_features = cf.predict_cars(img_cs, ret_features = True)
    t2 = time.time()

    data_to_save = {'windows':car_windows, 'features':window_features}

    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    output_filename = './{}/{}.p'.format(output_folder, basename)
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print('Frame {}, time taken {:.3f} secs. Saved to {}'.format(index, t2 - t1, output_filename))
