import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import glob
import time
from scipy.ndimage.measurements import label
import threading
import matplotlib.pyplot as plt
from lesson_functions import *
from boxes import *

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

class CarDetector(threading.Thread):
    def __init__(self, img_cs, boxes, global_box_no):
        threading.Thread.__init__(self)
        self.boxes = boxes
        self.img_cs = img_cs
        self.car_boxes = []
        self.car_box_nos = []
        self.global_box_no = global_box_no
        
    def run(self):
        for box_no, box in enumerate(self.boxes):
            if has_car(self.img_cs, box):
                self.car_boxes.append(box)
                self.car_box_nos.append(self.global_box_no + box_no)

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


thread_boxes = []

def detect_cars(img, all_boxes = False):
    global thread_boxes
    img_cs = cv2.cvtColor(img, color_space_code)

    if len(thread_boxes) == 0:
        boxes = get_all_boxes(img_cs)

        thread_boxes = []
        per_thread = len(boxes) // 1

        for i in range(0, len(boxes), per_thread):
            if i + per_thread < len(boxes):
                #print(i,i + per_thread)
                box_range = boxes[i:i + per_thread]
            else:
                #print(i,' Till End')
                box_range = boxes[i:]
            thread_boxes.append((box_range, i))

    threads = []
    for box_range, box_no in thread_boxes:
        thread = CarDetector(img_cs, box_range, box_no)
        threads.append(thread)
        thread.start()

    car_boxes = []
    for thread in threads:
        thread.join()
        car_boxes.extend(thread.car_boxes)
    
    return car_boxes
# def detect_cars(img, all_boxes = False):
#     img_cs = cv2.cvtColor(img, color_space_code)
    
#     boxes = get_all_boxes(img_cs)

#     print(len(boxes))
#     car_boxes = []
#     notcar_boxes = []
    
#     for box in boxes:
#         if has_car(img_cs, box):
#             car_boxes.append(box)
#         else:
#             notcar_boxes.append(box)
            
#     if not all_boxes:
#         return car_boxes
#     else:
#         return car_boxes, notcar_boxes

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

def draw_car_notcar(img, car_boxes, notcar_boxes):
    for box in notcar_boxes:
        cv2.rectangle(img, box[0], box[1], (255,0,0), 4)
    for box in car_boxes:
        cv2.rectangle(img, box[0], box[1], (0,255,0), 6)

# def block_from_pix(pix, n_blocks):
#     cell_no = pix // pix_per_cell
#     block_no = cell_no // cell_per_block
    
#     if block_no > n_blocks:
#         block_no = n_blocks - 1
#     return block_no

# def get_hog_dimensions(width, height, pixel_per_cell = 16, cell_per_block = 2, orient = 12):
#     n_cells_x = width // pix_per_cell
#     n_cells_y = height // pix_per_cell
#     n_blocks_x = n_cells_x - cell_per_block + 1
#     n_blocks_y = n_cells_y - cell_per_block + 1
#     return (n_blocks_y, n_blocks_x, cell_per_block, cell_per_block, orient)

# def subsample(features_org, boxes):
#     hog_64 = get_hog_dimensions(64,64,pix_per_cell,cell_per_block,orient)
#     hog_length_64 = hog_64[0] * hog_64[1] * hog_64[2] * hog_64[3] * hog_64[4]

# #     print('Original hog length:', hog_length_64)
#     pix_per_block = cell_per_block * cell_per_block * pix_per_cell
# #     print('Pixels per block:', pix_per_block)
    
#     sub_hogs = []
    
#     for box in boxes:
#         pix = []
        
#         sub_w = box[1][0] - box[0][0]
#         sub_h = box[1][1] - box[0][1]

#         sub_n_cells_x = sub_w // pix_per_cell
#         sub_n_cells_y = sub_h // pix_per_cell
#         sub_n_blocks_x = sub_n_cells_x - cell_per_block + 1
#         sub_n_blocks_y = sub_n_cells_y - cell_per_block + 1
        
# #         print('Box:', box)
# #         print('Block Start Y:', y1)
# #         print('Block Start X:', x1)
# #         print('Block End y:', y2)
# #         print('Block End x:', x2)
# #         print('Box # of blocks_x:', sub_n_blocks_x)
# #         print('Box # of blocks_y:', sub_n_blocks_y)
        
#         start_block_x = block_from_pix(box[0][0], features_org.shape[1])
#         start_block_y = block_from_pix(box[0][1], features_org.shape[0])
#         end_block_x = start_block_x + sub_n_blocks_x 
#         end_block_y = start_block_y + sub_n_blocks_x 
  
#         sub_hog = features_org[start_block_y:end_block_y, start_block_x:end_block_x,:,:,:]
# #         print(sub_hog.shape)
        
#         pixels = sub_hog.ravel()
#         cols = pixels.shape[0] / hog_length_64

# #         print('Raveled sub hog: ', pixels.shape)
# #         print('Number of new cols:', cols)
        
#         if (pixels.shape[0] % hog_length_64) > 0:
# #             print('Need to make data divisble by {}} so adding 0s'.format(hog_length_64))
#             needed = hog_length_64 * math.ceil(pixels.shape[0] / hog_length_64)
#             extra_required = needed - pixels.shape[0] 
#             pixels = np.hstack((pixels, np.zeros(extra_required)))
# #             print('after adding 0s, shape is', pixels.shape)

#         cols = pixels.shape[0] // hog_length_64
        
#         sub_pixels = pixels.reshape(-1, cols)
#         box_hog = np.mean(sub_pixels, axis=1)
        
#         sub_hogs.append(box_hog)
        
#     return sub_hogs
