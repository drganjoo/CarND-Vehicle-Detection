import time
import cv2
from skimage.feature import hog
from boxes import *
import threading
from lesson_functions import *

# orient = 11
# pix_per_cell = 16
# cell_per_block = 2
# spatial_feat = True
# hist_feat = True
# hog_feat = True
# color_space='LAB'
# hog_channel = [0,1,2]
# spatial_size = (32,32)
# hist_bins = 32

# class CarDetector(threading.Thread):
#     def __init__(self, img_cs, boxes):
#         threading.Thread.__init__(self)
#         self.boxes = boxes
#         self.img_cs = img_cs
#         self.combined_box_hogs = []
        
#     def run(self):
#         for box in self.boxes:
#             box_img = img_cs[box[0][1]:box[1][1], box[0][0]:box[1][0]]
#             box_img_reduce = cv2.resize(box_img, (64,64))

#             box_hog = []
#             for channel in hog_channel:
#                 features = hog(box_img_reduce[:,:,channel], 
#                                orientations=orient, 
#                                pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                cells_per_block=(cell_per_block, cell_per_block), 
#                                transform_sqrt=True, 
#                                visualise=False, 
#                                feature_vector=True,
#                                block_norm = 'L2-Hys')

#                 box_hog.append(features)
#             self.combined_box_hogs.append(np.ravel(box_hog))
                            
# img = cv2.imread('./project_video-frames/0000.jpg')
# img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# hog_channel = [0,1,2]
# boxes = get_all_boxes(img_cs)
# full_hog = []

# t1 = time.time()

# threads = []
# per_thread = len(boxes) // 20

# for i in range(0, len(boxes), per_thread):
#     if i + per_thread < len(boxes):
#         box_range = boxes[i:i + per_thread]
#     else:
#         box_range = boxes[i:]

#     thread = CarDetector(img_cs, box_range)
#     threads.append(thread)
#     thread.start()

# for thread in threads:
#     thread.join()
#     full_hog.extend(thread.combined_box_hogs)
    
# t2 = time.time()
# print('Total time taken for {} boxes to be reduced and hogged {:.2f} secs'.format(len(boxes), t2 - t1))

nbins = 32
bins_range = (0,256)

img = cv2.imread('./project_video-frames/0000.jpg')
img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
boxes = get_all_boxes(img_cs)

t1 = time.time()

    
for box_no, box in enumerate(boxes):
    box_img = img_cs[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    box_img_64 = cv2.resize(box_img, (64,64))    
    box_features = []

    img = box_img_64
    hist_features = color_hist(img, nbins=32)

    #box_features.append(np.ravel(hist_features))
    # channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    # channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    # channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

t2 = time.time()
print('Total time taken for {} boxes to be reduced and hogged {:.2f} secs'.format(len(boxes), t2 - t1))