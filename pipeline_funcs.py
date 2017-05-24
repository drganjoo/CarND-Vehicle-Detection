import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import glob
import time
from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt
from lesson_functions import *

svc = None
X_train_scaler = None

with open('svm.p', 'rb') as f:
    data = pickle.load(f)
    print(data)
    svc = data['svm']
    X_train_scaler = data['scaler']
    orient = data['orient'] 
    pix_per_cell = data['pix_per_cell']
    cell_per_block = data['cell_per_block']
    spatial_feat = data['spatial_feat']
    hist_feat = data['hist_feat']
    hog_feat = data['hog_feat']
    hog_channel = data['hog_channel']
    color_space = data['color_space']
    spatial_size = data['spatial_size']

print('Using Space Conversion: cv2.COLOR_RGB2' + color_space)
color_space_code = eval('cv2.COLOR_RGB2' + color_space)

def get_box_features(box_img_cs):
    if box_img_cs.shape[0] != 64 or box_img_cs.shape[1] != 64:
        raise Exception("Image has to be 64 x 64")
        
    features = extract_feature_image(box_img_cs, orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat = spatial_feat,
                               hist_feat = hist_feat,
                               spatial_size = spatial_size,
                               hog_feat = hog_feat)
    
    # normalize features of the box using the same parameters as were used while training
    return X_train_scaler.transform([np.ravel(features)])

def get_box_pixels(img, box):
    box_img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    box_img_64 = cv2.resize(box_img, (64,64))    
    return box_img_64

def has_car(img_cs, box):
    box_img_64 = get_box_pixels(img_cs, box)
    features = get_box_features(box_img_64)
    return svc.predict(features)

def medium_box(img, draw_color = None, no_of_boxes = None, offset = (-0.4, 0)):
    #x_start = 660
    x_start = 0
    x_stop = img.shape[1]
    y_start = 400
    y_stop = 600
    #box=(175,160)
    box=(11*16, 11*16)

    return get_boxes(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          box=box, 
          draw_color = draw_color, 
          no_of_boxes = no_of_boxes,
          offset_factor = offset)
    
def small_box(img, draw_color = None, no_of_boxes = None, offset = (-0.3, 0.5)):
    #x_start = 600
    x_start = 0
    x_stop = img.shape[1]
    y_start = 390
    y_stop = 550
    #box=(160,120)
    box=(7*16, 7*16)

    return get_boxes(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          box=box, 
          draw_color=draw_color, 
          no_of_boxes = no_of_boxes,
          offset_factor = offset)


def smallest_box(img, draw_color = None, no_of_boxes = None, offset = (-0.2, 0.5)):
    #x_start = 600
    # box = (80,70)
    box=(5*16, 5*16)
    x_start = 20
    x_stop = img.shape[1] - 20
    y_start = 400
    y_stop = y_start + 70 + 40

    return get_boxes(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          box=box, 
          draw_color = draw_color, 
          no_of_boxes = no_of_boxes,
          offset_factor=offset)

def tiny_box(img, draw_color = None, no_of_boxes = None, offset = (-0.5, 0.5)):
    #x_start = 600
    x_start = 120
    x_stop = img.shape[1] - 120
    y_start = 300
    y_stop = 450
    box=(64,64)

    return get_boxes(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          box=box, 
          draw_color = draw_color, 
          no_of_boxes = no_of_boxes,
          offset_factor=offset)

def get_all_boxes(img):
    boxes = []
    #boxes.extend(big_box(img))
    boxes.extend(medium_box(img))
    boxes.extend(small_box(img))
    boxes.extend(smallest_box(img))
    #boxes.extend(tiny_box(img))
    return boxes

def detect_cars(img, all_boxes = False):
    img_cs = cv2.cvtColor(img, color_space_code)
    
    boxes = get_all_boxes(img_cs)
    print(len(boxes))
    car_boxes = []
    notcar_boxes = []
    
    for box in boxes:
        if has_car(img_cs, box):
            car_boxes.append(box)
        else:
            notcar_boxes.append(box)
            
    if not all_boxes:
        return car_boxes
    else:
        return car_boxes, notcar_boxes

def heatmap_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] =  0
    
def add_heat(heatmap, boxes):
    for box in boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
def show_heatmap(heatmap):
    heatmap_red = np.dstack(((heatmap * 255/10), np.zeros_like(heatmap), np.zeros_like(heatmap)))
    plt.imshow(heatmap_red)
    
def draw_labeled_bboxes(img, labels):
    colors = [(255,0,0), (0,255,0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (255, 150, 0), (255, 0, 150), (150, 75, 0), (150, 0, 75)]

    labels_img = np.zeros_like(img)
    
    # add more colors to the array in case there are more labels than number of defined colors
    if labels[1] == len(colors):
        extra = labels[1] - len(colors)
        colors.extend([(255,0,0)] * extra)
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        mask = labels[0] == car_number
        labels_img[mask] = colors[car_number - 1]
        
        nonzero = (mask).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return labels_img

def get_heat_img(img, heatmap):
    hot_colors = [(90,0,0), (212,0,0), (255, 63, 0), (255,103,0), (255,225,0), (255,225,0), (255,225,0)]

    heat_clipped = np.clip(heatmap, 0, 255)
    max_heat = np.max(heat_clipped)
    if max_heat >= len(hot_colors):
        heatmap = (heatmap / max_heat * len(hot_colors)).astype(np.int_)
    
    heat_img = np.zeros(shape=(heatmap.shape[0], heatmap.shape[1], 3)).astype(np.uint8)
    
    for index in range(0, len(hot_colors)):
        locations = np.where(heatmap == index + 1)
        heat_img[locations[0], locations[1]] = hot_colors[index]
    
    return cv2.addWeighted(img, 0.4, heat_img, 0.6, 0)
    #return heat_img

def draw_car_notcar(img, car_boxes, boxes):
    for box in notcar_boxes:
        cv2.rectangle(img, box[0], box[1], (255,0,0), 4)
    for box in car_boxes:
        cv2.rectangle(img, box[0], box[1], (0,255,0), 6)