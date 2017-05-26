import matplotlib.pyplot as plt
import cv2, os, time, pickle, threading
import numpy as np
from lesson_functions import *
from windows import *
from collections import deque
from scipy.ndimage.measurements import label


colors = [(255,0,0), (0,255,0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255,255,255),
            (255, 0, 255), (255, 150, 0), (255, 0, 150), (150, 75, 0), (150, 0, 75)]
        
class SingleFrameCarFinder():
    def __init__(self, svc, X_scaler, params_for_feature):
        global params
        params = params_for_feature

        self.svc = svc
        self.X_scaler = X_scaler
        self.windows = None
        self.img_cs = None
        self.car_windows = []
        self.heatmap = None
        self.params = params_for_feature
    
    def init_for_frame(self, img_cs):
        self.img_cs = img_cs
        self.img_rgb = self.colorspace2rgb(self.img_cs)
        self.heatmap = np.zeros(img_cs.shape[:2], np.float)
        self.car_windows = []

    def colorspace2rgb(self, img_cs):
        return cv2.cvtColor(img_cs, eval('cv2.COLOR_' + self.params['color_space'] + '2RGB'))

    def rgb2colorspace(self, img_rgb):
        return cv2.cvtColor(img_rgb, eval('cv2.COLOR_RGB2' + self.params['color_space']))

    def find_features_for_windows(self, windows):
        window_features = []

        for w in windows:
            window_img = self.img_cs[w[0][1]:w[1][1], w[0][0]:w[1][0]]
            window_img_64 = cv2.resize(window_img, (64,64))

            w_features = single_img_features(window_img_64, 
                        spatial_size=self.params['spatial_size'],
                        hist_bins=self.params['hist_bins'], 
                        orient=self.params['orient'], 
                        pix_per_cell=self.params['pix_per_cell'], 
                        cell_per_block=self.params['cell_per_block'],
                        hog_channel=self.params['hog_channel'],
                        spatial_feat=self.params['spatial_feat'],
                        hist_feat=self.params['hist_feat'], 
                        hog_feat=self.params['hog_feat'],
                        vis=False)
        
            window_features.append(w_features)
        
        return self.X_scaler.transform(window_features)

    def find_features(self):
        if self.windows is None:
            self.windows = get_all_windows(self.img_cs)
        
        return self.find_features_for_windows(self.windows)

    def draw_windows(self, img = None, windows = None, color=(0, 255, 0)):
        #Tracer()() #this one triggers the debugger
        
        if img is None:
            img = np.copy(self.img_rgb)
            
        if windows is None:
            windows = self.car_windows
            
        for window in windows:
            cv2.rectangle(img, window[0], window[1], color, 4)
        return img

    def add_heat(self, windows, heatmap = None):
        if heatmap is None:
            heatmap = self.heatmap

        for window in windows:
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

    def get_heat_img(self, heatmap = None, img = None, weight = (0.4,0.6)):
        if heatmap is None:
            heatmap = self.heatmap
        if img is None:
            img = self.img_rgb
            
        hot_colors = [(90,0,0), (212,0,0), (255, 63, 0), (255,103,0), (255,225,0), (255,225,0), (255,225,0)]

        heatmap_clipped = np.clip(heatmap, 0, 255)
        
        max_heat = np.max(heatmap_clipped)
        cap = len(hot_colors)
        
        if max_heat >= cap:
            heatmap_clipped = (heatmap_clipped / max_heat) * cap

        heat_img = np.zeros(shape=self.img_rgb.shape).astype(np.uint8)

        for index, color in enumerate(hot_colors):
            heat_img[heatmap_clipped > index] = color

        return cv2.addWeighted(img, weight[0], heat_img, weight[1], 0)
        
    def car_exists(self, window_feature):
        return self.svc.predict([window_feature])

    def predict_cars(self, img_cs, ret_features = False, car_windows = None, window_features = None):
        self.init_for_frame(img_cs)

        if car_windows is None or window_features is None:
            window_features = self.find_features()

            self.car_windows = []
            for index, w_feature in enumerate(window_features):
                if self.car_exists(w_feature):
                    self.car_windows.append(self.windows[index])
        else:
            window_features = window_features
            self.car_windows = car_windows

        self.add_heat(self.car_windows)
        
        if not ret_features:
            return self.car_windows
        else:
            return self.car_windows, window_features

# img_cs = load_image('./project_video-frames/0000.jpg', params['color_space'])
# cf = SingleFrameCarFinder(svc, X_scaler)
# car_windows = cf.predict_cars(img_cs)

# print('Cars found: {}'.format(len(car_windows)))

# cars_found = cf.draw_windows(windows = car_windows)
# heat_img = cf.get_heat_img()

# f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,10))
# ax1.imshow(cars_found)
# ax2.imshow(heat_img)

# plt.show()


def debug(str, *args):
    #print(str, args)
    pass

def get_w_h(box):
    width = box[1][0] - box[0][0]
    height = box[1][1] - box[0][1]
    return width, height

def get_center(box):
    width, height = get_w_h(box)
    return np.int_([box[0][0] + width // 2, box[0][1] + height // 2])

def heatmap_threshold(heatmap, threshold):
    heatmap[heatmap < threshold] =  0

RIGHT_THRESHOLD = 10
LEFT_THRESHOLD = 20
MAX_MISSFRAMES = 10
MIN_FRAMES_TO_CENTER_SEARCH = 5
FRAME_COUNT_MAYBE_CARS = 10
MAX_FRAME_COUNT_MAYBE_CARS = 10

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
        self.miss_frames = 0
        self.accounted_miss_frames = 0
        self.frame_no_when_seen = 0 # which frame did we see this car

    def is_box_enclosing(self, box):
        return (self.center[0] >= box[0][0]) and (self.center[0] <= box[1][0])                 and (self.center[1] >= box[0][1]) and (self.center[1] <= box[1][1])
    
    def might_go_out(self):
        # in case the car is towards the bottom of the screen and is not a new car
        return self.frame_count > 3 and self.center[1] >= 650 
        
    def update_box(self, box):
        self.miss_frames = 0
        self.accounted_miss_frames = 0
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

            center = get_center(box)
            
            last_center = self.center
            self.center = last_center * (1 - self.center_alpha) + center * self.center_alpha
            
            self.bounding_box = (int(self.center[0] - new_width // 2), int(center[1] - new_height // 2)),                                 (int(center[0] + new_width // 2), int(center[1] + new_height // 2))
        
            delta = self.center - last_center
            self.speed += delta

    def nudge_as_per_speed(self):
#         delta_per_frame = self.speed / self.frame_count
#         estimated_movement = (self.miss_frames - self.accounted_miss_frames) / delta_per_frame
#         self.accounted_miss_frames = self.miss_frames
#         self.center[0] += estimated_movement[0]
#         self.center[1] += estimated_movement[1]
        pass

    def get_all_fit_boxes(self, img):
        # will return all box sizes with our current center, used for rechecking to make sure the car is no
        # longer in frame
        all_box_sizes = get_all_window_sizes(img)

        detection_boxes = []
        for width, height in all_box_sizes:
            bounding_box = self.make_box_from_center(width, height)
            detection_boxes.append(bounding_box)

        return detection_boxes

    def make_box_from_center(self, width, height):
        bounding_box = ((max(int(self.center[0] - width // 2),0),
                                 max(int(self.center[1] - height // 2), 0)),
                                 (min(int(self.center[0] + width // 2), 1280),
                                 min(int(self.center[1] + height // 2), 720)))
        return bounding_box
            
class VehicleIdentifier():
    def __init__(self, params_for_feature):
        self.min_frames = 20
        self.frames_to_avg = 10
        self.heatmaps = deque(maxlen = self.frames_to_avg)
        self.cars = []
        self.cars_gone_off = []
        self.maybe_cars = []
        self.last_frame_no = 0  # helps in debugging, pickle object and then start again from same frame
        self.car_finder = SingleFrameCarFinder(svc, X_scaler, params_for_feature)
        self.car_windows = None
        self.add_windows = None # these are the ones that are re-created when we do not see a car
        self.combined_heatmap = []
        self.combined_heatmap_nonthreshold = []
        self.img_cs = None
        self.img_rgb = None
        self.add_windows = None
        
    def get_bounding_box(self, labels, car_no):
        mask = labels[0] == car_no

        nonzero = (mask).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        return bbox

    def get_car_for_bbox(self, bbox, which_cars = None):
        car = None
        center = get_center(bbox)
        
        if which_cars is None:
            which_cars = self.cars

        for c in which_cars:
            #if c.is_within_bounds(center):
            if (c.center[0] >= bbox[0][0]) and (c.center[0] <= bbox[1][0]) and (c.center[1] >= bbox[0][1]) and (c.center[1] <= bbox[1][1]):
                car = c
                break
                
        return car
    
    def draw_labels_img(self, labels):
        labels_img = np.zeros_like(self.img_rgb)

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
            color_index = car.car_no % len(colors)
            cv2.rectangle(output_img, car.bounding_box[0], car.bounding_box[1], colors[color_index], 6)

    def make_bounding_box(self, boxes):
        if len(boxes) < 2:
            nboxes = np.array([boxes])
        else:
            nboxes = np.array(boxes)
        # change the array shape so that it becomes
        # x = [[1,2,3,4],[4,5,6,7]]
        nboxes = nboxes.reshape(nboxes.shape[0], -1)
        x1 = np.min(nboxes[:,0])
        y1 = np.min(nboxes[:,1])
        x2 = np.max(nboxes[:,2])
        y2 = np.max(nboxes[:,3])

        box = (x1,y1),(x2,y2)
        return box

    def figure_out_cars(self, windows):
        self.combined_heatmap = np.sum(self.heatmaps, axis=0)
        #self.combined_heatmap = np.average(self.heatmaps, axis=0)
        self.combined_heatmap_nonthreshold = np.copy(self.combined_heatmap)

        #print(np.max(summed_up))

        heatmap_threshold(self.combined_heatmap[:,600:], RIGHT_THRESHOLD)
        heatmap_threshold(self.combined_heatmap[:,:600], LEFT_THRESHOLD)
        
        self.labels = label(self.combined_heatmap)
        
        # create a dictionary for all cars we know off
        # check which ones are not here in this frame and mark their miss_frame count
        cars_seen_this_frame = {}

        for car in self.cars:
            cars_seen_this_frame[car] = False

        # gather all car boxes for this frame
        car_boxes = []
        for car_no in range(1, self.labels[1]+1):
            car_boxes.append(self.get_bounding_box(self.labels, car_no))

        # identify new cars and remove old ones
        for car_box in car_boxes:
            car = self.get_car_for_bbox(car_box)
            
            if car is None:
                print('There is a possible car at box: {}, lets look for an already maybe_car we might have'.format(car_box))

                # Check in case we already have this bounding box in a maybe_car.
                # Increment frame connt and see if it is to be moved to a car now
                car = self.get_car_for_bbox(car_box, which_cars = self.maybe_cars)

                if car is not None:
                    print('Ok we found a maybe car, lets see if this one was recent or very old')

                    no_of_frames_passed = self.last_frame_no - car.frame_no_when_seen 
                    
                    # in case the car we found in the maybe list is very old and is a zoombie
                    # don't consider it the same car at all
                    
                    if no_of_frames_passed > MAX_FRAME_COUNT_MAYBE_CARS:
                        print('An old maybe car, earlier seen at {} found, resurrecting it but still a maybe not actual'.format(car.frame_no_when_seen))

                        car.frame_no_when_seen = self.last_frame_no
                        car.frame_count = 1
                        car.bounding_box = car_box
                    else:
                        print('Recent may_be, last seen {} frames ago. Total frames: {} Total Required: {} .'.format(no_of_frames_passed, car.frame_count, FRAME_COUNT_MAYBE_CARS))

                        car.frame_count += 1
                        
                        # there should be at least 5  frames that the car is visible for and
                        # it should be visible at least in 50% of the last frames
                        if (car.frame_count > FRAME_COUNT_MAYBE_CARS) and (car.frame_count / no_of_frames_passed > 0.5):
                            w,h = get_w_h(car_box)
                            if (w > 20) and (h > 20):
                                self.cars.append(car)
                                cars_seen_this_frame[car] = True  # add a new car to the cars_seen array

                                print('Remove from maybe cars and adding it to the active car list')
                                self.maybe_cars.remove(car)
                            else:
                                print('Too small to be a car')
                else:
                    # check how close are we to another car, may be its a  false
                    # bounding box from labels, we will add it to the possible cars array
                    # and will see if it is still there after some frames or not

                    if (len(self.cars) > 1) and (not self.very_close_to_other(car_box)):
                        car = Vehicle(car_no)
                        car.frame_no_when_seen = self.last_frame_no

                        self.cars.append(car)
                        cars_seen_this_frame[car] = True  # add a new car to the cars_seen array
                    else:
                        car = Vehicle(car_no)
                        car.frame_no_when_seen = self.last_frame_no
                        
                        self.maybe_cars.append(car)
            else:
                cars_seen_this_frame[car] = True

            # may be it was a false prediction from labels and we are not going to add this car
            if car is not None:
                # maybe label is confused and giving us a rectangle that is either merged
                # or very close to some other car, stick to our update DO NOT believe what
                # label is saying
                if (car.bounding_box is not None) and self.very_close_to_other(car_box):
                    # how big or small is the new box compared to our old box
                    cb_w, cb_h = get_w_h(car_box)
                    car_w, car_h = get_w_h(car.bounding_box)

                    delta_h = np.abs(car_w - cb_w)
                    delta_w = np.abs(car_h - cb_h)

                    if (delta_h / car_h < 0.5) and (delta_w / car_w < 0.5):
                        car.update_box(car_box)
                else:
                    car.update_box(car_box)
            
        # is there a car that we did not see this time? if yes, maybe
        # it could have been overlapping
        for car in cars_seen_this_frame:
            if cars_seen_this_frame[car] == False:
                print('Did not see car #', car.car_no, ' speed:', car.speed)
                
                # check if there were any boxes in which this car was detected?
                car_windows = []
                for window in windows:
                    if car.is_box_enclosing(window):
                        car_windows.append(window)

                if len(car_windows) > 0:
                    print('Looks like there were some windows')

                    # make a local heatmap, find an enclosing box around it and then
                    # use that to update the car bounding box
                    heatmap = np.zeros(shape=self.img_cs.shape[:2])
                    self.car_finder.add_heat(car_windows, heatmap)

                    heatmap /= np.max(heatmap)
                    heatmap[heatmap < 0.7] = 0
                    zy, zx = heatmap.nonzero()
                    left = np.min(zx)
                    right = np.max(zx)
                    top = np.min(zy)
                    bottom = np.max(zy)

                    #bbox = self.make_bounding_box(car_windows)
                    bbox = ((left, top), (right,bottom))
                    car.update_box(bbox)
                    cars_seen_this_frame[car] = True
                else:
                    cars_seen_this_frame[car] = self.car_gone_out(car)

        # remove all cars that were not seen
        for index, car in enumerate(self.cars):
            if not cars_seen_this_frame[car]:
                self.cars_gone_off.append(car)
                del self.cars[index]

        # # check maybe cars might need to be kicked out now
        # if len(self.maybe_cars) > 0:
        #     copy_maybe_cars = self.maybe_cars
        #     self.maybe_cars = []
        #     for car in copy_maybe_cars:
        #         if self.last_frame_no - (car.frame_no_when_seen + car.frame_count) < FRAME_COUNT_MAYBE_CARS:
        #             self.maybe_cars.append(car)

    def very_close_to_other(self, box):
        very_close = False

        y_closeness_x = [(475,125), (300,20), (0, 10)]

        for car in self.cars:
            car_leftx = car.bounding_box[0][0]
            car_lefty = car.bounding_box[1][1]
            car_rightx = car.bounding_box[1][0]
            car_righty = car.bounding_box[1][1]

            if box[1][0] < car_leftx:
                # to the left of this car
                distance_x = car_leftx - box[1][0]

                what_is_close = 0

                for y_based_distance in y_closeness_x:
                    if car_lefty > y_based_distance[0]:
                        what_is_close = y_based_distance[1]
                        break

                if distance_x < what_is_close:
                    # now we need to check how close is it in Y
                    #distance_yb = car.bounding_box[1][1] - box[1][1]
                    very_close = True
                    break
            elif box[1][0] > car_rightx:
                # to the left of this car
                distance_x = box[1][0] - car_rightx

                what_is_close = 0

                for y_based_distance in y_closeness_x:
                    if car_righty > y_based_distance[0]:
                        what_is_close = y_based_distance[1]
                        break

                if distance_x < what_is_close:
                    # now we need to check how close is it in Y
                    #distance_yb = car.bounding_box[1][1] - box[1][1]
                    very_close = True
                    break
                
        return very_close


    # def remove_overlapping(self, cars, i, j):
    #     # check if there is an overlap or not
    #     box1 = cars[i]
    #     box2 = cars[2]

    #     if box1[1][0] > box2[0][0]:
    #         box1


    # a car that used to be in the frame is no longer with us, lets
    # try to create a new window that is centered at the last point we 
    # knew for the car and then see may be it is still there
    def car_gone_out(self, car):
        car.miss_frames += 1

        if car.miss_frames > MAX_MISSFRAMES:
            return False

        # we did estiamte that the car would be going out so its ok if it is no longer
        # in frame
        if car.might_go_out():
            return False

        # the car should have been with us for long before we do a center search on it
        if car.frame_count < MIN_FRAMES_TO_CENTER_SEARCH:
            return False

        print('Car gone out, it has been with us for {} frames'.format(car.frame_count))

        # lets resample around the box we have for the car and then see if we
        # can detect a car there

        centered_windows = car.get_all_fit_boxes(self.img_cs)
        centered_windows_features = self.car_finder.find_features_for_windows(centered_windows)

        car_windows = []
        for index, cw_features in enumerate(centered_windows_features):
            if self.car_finder.car_exists(cw_features):
                cw = car_windows[index]
                car_windows.append(cw)

        car_seen = False                
        if len(car_windows) > 0:
            # okay we found some car using some of our box sizes, so lets make a new bounding box
            # using the average of all boxes
            avg_w = 0
            avg_h = 0

            for window in car_windows:
                w, h = get_w_h(window)
                avg_w += w
                avg_h += h

            avg_w /= len(car_windows)
            avg_h /= len(car_windows)

            box = car.make_box_from_center(avg_w, avg_h)
            car.update_box(box)
            car_seen = True

            self.add_windows = car_windows

        return car_seen

    def overlay_images(self):
        output_img = np.copy(self.img_rgb)

        img_tl = np.copy(self.img_rgb)
        self.car_finder.draw_windows(img = img_tl, color = (0, 255, 0))
        if self.add_windows is not None:
            self.car_finder.draw_windows(img = img_tl, color = (255, 0, 0), windows = self.add_windows)

        # overlay image heat on this frame
        img_tl = self.car_finder.get_heat_img(img=img_tl, weight=(0.5,0.5))
        
        # draw this frame's heatmap on the top center of the image
        output_img[0:180,0:320] = cv2.resize(img_tl, (320, 180))

        # wait till we have enough heat maps to average out
        #print('Heatmaps accumulated: {}, we wait for {}'.format(len(self.heatmaps), self.frames_to_avg))

        if len(self.heatmaps) < self.frames_to_avg:
            cv2.putText(output_img, "Waiting for frames", (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            #print('Enough heatmaps')
        
            self.draw_cars(output_img)

            labels_img = self.draw_labels_img(self.labels)
            heat_img_nonthreshold = self.car_finder.get_heat_img(self.combined_heatmap_nonthreshold)
            heat_img = self.car_finder.get_heat_img(self.combined_heatmap)

            output_img[0:180,320:640] = cv2.resize(heat_img_nonthreshold, (320, 180))
            output_img[0:180,640:960] = cv2.resize(heat_img, (320, 180))
            output_img[0:180,960:1280] = cv2.resize(labels_img, (320,180))

            cv2.putText(output_img, "This Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.putText(output_img, "{} Heatmaps".format(self.frames_to_avg), (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.putText(output_img, "Thresholded", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            cv2.putText(output_img, "Cars: {}".format(len(self.cars)), (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        return output_img

    def process_frame(self, img_rgb, car_windows = None, window_features = None):
        debug('Processing image, len(car)', len(self.cars))
        
        self.add_windows = None
        self.img_rgb = img_rgb
        self.img_cs = self.car_finder.rgb2colorspace(img_rgb)
        
        self.car_windows = self.car_finder.predict_cars(self.img_cs, car_windows = car_windows, window_features = window_features)

        self.heatmaps.append(self.car_finder.heatmap)
        
        if len(self.heatmaps) >= self.frames_to_avg:
            self.figure_out_cars(self.car_windows)

        return self.overlay_images()


if __name__ == '__main__':
    with open('svm-yuv-16x2x11.p', 'rb') as f:
        data = pickle.load(f)
        svc = data['svm']
        X_scaler = data['scaler']
        params = data['params']
        
    print('Data Loaded')
    print('SVC:', svc)
    print('Params', params)
    # car = Vehicle(1)
    # car.center = (500,500)
    # print(car.make_box_from_center(100,100))

    # filename = './project_video-frames/0000.jpg'
    # img_rgb = load_image(filename, 'RGB')

    # vi = VehicleIdentifier()
    # overlay_img = vi.process_frame(img_rgb)

    # plt.imshow(overlay_img)
    # plt.show()

    def get_frame_filenames(index):
        filename = './project_video-frames/{:04d}.jpg'.format(index)
        window_filename = './project_video-frames/car_windows/{:04d}.p'.format(index)

        if not os.path.exists(window_filename):
            window_filename = None
        
        return filename, window_filename

        
    id_filename = './data/identifier.p'

    if os.path.exists(id_filename):
        with open(id_filename, 'rb') as f:
            try:
                identifier = pickle.load(f)
                print('Old identifier loaded with frame:', identifier.last_frame_no)
            except EOFError as e:
                print(e)
                print('Loading a new identifier')
                identifier = VehicleIdentifier(params)
                identifier.threshold = 20
    else:
        identifier = VehicleIdentifier(params)
        print('New identifier created')


    for identifier.last_frame_no in range(identifier.last_frame_no + 1, identifier.last_frame_no + 400):
        #filename = './project_video-frames/{:04d}.jpg'.format(identifier.last_frame_no)
        filename, window_filename = get_frame_filenames(identifier.last_frame_no)
        
        # read RGB since thats what video will give us and then our function
        # internally converts it to LAB
        img = load_image(filename, 'RGB')
        
        # load window data and features if that has been saved
        t1 = time.time()
        if window_filename is not None:
            with open(window_filename, 'rb') as f:
                window_data = pickle.load(f)
                car_windows = window_data['windows']
                window_features = window_data['features']

        output_img = identifier.process_frame(img, car_windows = car_windows, window_features = window_features)
        t2 = time.time()
        print('Frame {}, time taken {:.3f} secs'.format(identifier.last_frame_no, t2 - t1))

        # plt.imshow(output_img)
        # plt.show()
        
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./frame-cars/{:04d}.jpg'.format(identifier.last_frame_no), output_img)

        with open(id_filename, 'wb') as f:
            pickle.dump(identifier, f)

        if (identifier.last_frame_no % 20) == 0:
            filename_20 = './data/identifier-{}.p'.format(identifier.last_frame_no)
            with open(filename_20, 'wb') as f:
                pickle.dump(identifier, f)

        # print('Frame:', frame_no)
        # plt.imshow(output_img)
        # plt.show(block=False)

    # save a copy of the identifier for next time
    # filename = './data/identifier-{}.p'.format(identifier.last_frame_no)
    # with open(filename, 'wb') as f:
    #     pickle.dump(identifier, f)
    #     print('Backup copy saved to:', filename)
