import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage.measurements import label
from lesson_functions import *
from pipeline_funcs import *
from collections import deque
import time

def get_w_h(box):
    width = box[1][0] - box[0][0]
    height = box[1][1] - box[0][1]
    return width, height

def get_center(box):
    width, height = get_w_h(box)
    return np.int_([box[0][0] + width // 2, box[0][1] + height // 2])

def debug(str, *args):
    #print(str, args)
    pass


class Vehicle():
    def __init__(self, car_no):
        self.bounding_box = None
        self.last_bounding_box = None
        self.speed = 0.0
        self.center = None
        self.car_no = car_no
        self.alpha = 0.5
        self.center_alpha = 0.7
        self.frame_count = 0
        self.bound_pixels = 20
        self.speed = 0
        self.miss_frames = 0

    def is_box_enclosing(self, box):
        w,h = get_w_h(box)
        bw, bh = get_w_h(self.bounding_box)
        w_ratio = np.abs(1 - bw / w)
        h_ratio = np.abs(1 - bh / h)

        allowed_range = (0.1, 0.1)
        return (w_ratio <= allowed_range[0] and h_ratio <= allowed_range[1])
    
    def might_go_out(self):
        # in case the car is towards the bottom of the screen and is not a new car
        return self.frame_count > 3 and self.center[1] >= 650 
    
    def is_within_bounds(self, center):
        # get the sum of square difference between 2 centers and see if it is 
        # within bounds then yes, otherwise no
        return np.sqrt(np.sum((self.center - center) ** 2)) <= self.bound_pixels
        
    def update_box(self, box):
        self.miss_frames = 0
        self.frame_count += 1

        width, height = get_w_h(box)
        
        if self.center is None:
            self.bounding_box = box
            self.center = get_center(box)
        else:
            self.last_bounding_box = self.bounding_box

            # increase / decrease slowly
            old_width, old_height = get_w_h(self.last_bounding_box)
            
            new_width = old_width * (1 - self.alpha) + width * self.alpha
            new_height = old_height * (1 - self.alpha) + height * self.alpha

            debug('Last bbox', self.last_bounding_box, 'New Box:', box)
            debug('Old W,h', old_width, old_height, 'New Width Height', new_width, new_height)

            center = get_center(box)
            debug('Center given', center)
            
            last_center = self.center
            self.center = self.center * (1 - self.center_alpha) + center * self.center_alpha
            
            self.bounding_box = (int(self.center[0] - new_width // 2), int(center[1] - new_height // 2)), \
                                (int(center[0] + new_width // 2), int(center[1] + new_height // 2))
        
            delta = self.center - last_center
            self.speed += delta
            
            debug('Filtered center', self.center)
            debug('Final BBox', self.bounding_box)
            
class VehicleIdentifier():
    def __init__(self):
        #self.fps = 25
        self.min_frames = 20
        self.threshold = 10
        self.frames_to_avg = self.min_frames
        self.heatmaps = deque(maxlen = self.frames_to_avg)
        self.cars = []
        self.max_miss_frames = 3
        
    def get_bounding_box(self, labels, car_no):
        mask = labels[0] == car_no

        nonzero = (mask).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        return bbox

    def get_car_for_bbox(self, bbox):
        car = None
        center = get_center(bbox)
        
        for c in self.cars:
            if c.is_within_bounds(center):
                car = c
                break
                
        return car
    
    def color_labels(self, img, labels):
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

        return labels_img

    def draw_cars(self, output_img):
        for car in self.cars:
            cv2.rectangle(output_img, car.bounding_box[0], car.bounding_box[1], (0,255,0), 6)

    def make_bounding_box(self, boxes):
        nboxes = np.array(boxes)
        nboxes = nboxes.reshape(nboxes.shape[0], -1)
        x1 = np.min(nboxes[:,0])
        y1 = np.min(nboxes[:,1])
        x2 = np.max(nboxes[:,2])
        y2 = np.max(nboxes[:,3])

        box = (x1,y1),(x2,y2)
        return box

    def figure_out_cars(self, boxes):
        summed_up = np.sum(self.heatmaps, axis=0)
        heatmap_threshold(summed_up, self.threshold)
            
        cars_seen = np.array([False] * len(self.cars))
        
        labels = label(summed_up)
        for car_no in range(1, labels[1]+1):
            debug('Label says car #:', car_no)
            car_box = self.get_bounding_box(labels, car_no)
            car = self.get_car_for_bbox(car_box)
            
            if car is None:
                debug('We have never seen this car # before. box:', car_box)
                car = Vehicle(car_no)
                self.cars.append(car)
            else:
                debug('Have seen this car before, it is car #', car.car_no)
                cars_seen[car.car_no - 1] = True
            
            car.update_box(car_box)
            
        # is there a car that we did not see this time? if yes, maybe
        # it could have been overlapping
        for i in range(len(cars_seen)):
            if cars_seen[i] == False:
                car = self.cars[i]
                debug('did not see car #', car.car_no, 'speed:', car.speed)
                # make sure bounding box has not been overlapped with some other's
                car_boxes = []
                for box in boxes:
                    if car.is_box_enclosing(box):
                        car_boxes.extend(box)
                
                if len(car_boxes) > 0:
                    bbox = self.make_bounding_box(car_boxes)
                    car.update_box(bbox)
                    cars_seen[i] = True
                else:
                    car.miss_frames += 1
                    if car.miss_frames < self.max_miss_frames:
                        cars_seen[i] = True
        
        # remove all cars that were not seen
        self.cars = self.cars[cars_seen]
        
        #labels_img = draw_labeled_bboxes(output_img, labels)
        labels_img = self.color_labels(output_img, labels)
        self.draw_cars(output_img)

        heat_img = get_heat_img(img, summed_up)
        output_img[0:180,640:960] = cv2.resize(heat_img, (320, 180))
        output_img[0:180,960:1280] = cv2.resize(labels_img, (320,180))

        cv2.putText(output_img, "Cars: {}".format(len(self.cars)), (970, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def process_image(self, img):
        debug('Processing image, len(car)', len(self.cars))
        
        #t1 = time.time()
        #print(color_space_code)
        boxes = detect_cars(img)

        #t2 = time.time()
        #print('Time in detection', t2-t1)

        #print(boxes)
        heatmap = np.zeros(img.shape[:2], np.float)
        add_heat(heatmap, boxes)
        self.heatmaps.append(heatmap)

        heat_img = get_heat_img(img, heatmap)
        output_img = np.copy(img)
        
        # draw boxes found in this frame on the image
        img_tl = np.copy(img)
        for box in boxes:
            cv2.rectangle(img_tl, box[0], box[1], (0,255,0), 4)
        output_img[0:180,0:320] = cv2.resize(img_tl, (320, 180))
        
        # draw this frame's heatmap on the top center of the image
        output_img[0:180,320:640] = cv2.resize(heat_img, (320, 180))

        # wait till we have enough heat maps to average out
        if len(self.heatmaps) >= self.frames_to_avg:
            debug('Heatmap can be summed')
            self.figure_out_cars(boxes)
        else:
            cv2.putText(output_img, "Waiting for frames", (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return output_img

identifier = VehicleIdentifier()

for frame_no in range(0, 100):
    filename = './project_video-frames/{:04d}.jpg'.format(frame_no)
    
    # read RGB since thats what video will give us and then our function
    # internally converts it to LAB
    img = load_image(filename, 'RGB')
    output_img = identifier.process_image(img)
    
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./frame-cars/{:04d}.jpg'.format(frame_no), output_img)
    
    # print('Frame:', frame_no)
    # plt.imshow(output_img)
    # plt.show(block=False)

with open('identifier.p', 'wb') as f:
    pickle.dump(identifier)