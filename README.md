# Vehicle Detection Project


[//]: # (Image References)
[distribution]: ./output_images/data-distribution.png
[augmented-not-cars]: ./output_images/augmented-not-cars.png
[cars-lab]: ./output_images/cars-lab.png
[notcars-lab]: ./output_images/notcars-lab.png
[car-hogs]: ./output_images/car-hogs.png
[notcar-hogs]:./output_images/notcar-hogs.png
[window-shapes]:./output_images/window_shapes.png
[sliding-window-hascar]:./output_images/sliding_window_hascar.png
[false-detection]:./output_images/false_detection.png
[heatmap]:./output_images/heatmap.jpg
[sample-missframe]:./output_images/0595.jpg
[sample-frame]: ./output_images/sample_frame.jpg
[combined-hogs]: ./output_images/combined-hogs.png
[various-options]: ./output_images/various-options.png
[false-got-through]: ./output_images/false-got-through.jpg
[good-false]: ./output_images/good-false.jpg
[bad-label]: ./output_images/bad-label.jpg

The project has been broken down into two parts, one is training the SVM and the other is processing the video / identifying cars.
## Project Video 

Project video is available at [./project_video_cars.mp4](./project_video_cars.mp4)

## Training SVM

The training part has been coded in a jupyter notebook, where as vehicle identification pipeline has been coded as python files.

[VehicleDetection-Training](training.ipynb)  ** Used for SVM training **
[window_testing](window_testing.ipynb) ** Different window size / heat map tests **  
[car_finder.py](car_finder.py)  ** Main car detection code that used thresholded heatmaps **
[Exploration](exploration.ipynb)  ** Various color spaces etc. that were tested **

Some helper files:

[windows.py](windows.py)  ** Definition of window sizes **
[lesson_functions.py](lesson_functions.py)  ** Misc functions from the lessons like get_hog_feature etc. **

### Data Used For Training

Mostly used GTI and KITTI car / not-car images that were given as part of [vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

During testing a lot of times the yellow lane line was being misclassified as a car, therefore I generated a number of smaller 64x64 images from the first 100 frames of the video, which had no car visible and used them as part of non-vehicles set.

I tried using the udacity's labeled data set [Udacity Labeled Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) but at that point I didn't know the column headings were wrong and I could hardly get the correct car frames out of it, so didn't use it at that point. [extract_udacity.ipynb](extract_udacity.ipynb) I later found out that the correct column labels were x_min, x_max, y_min and y_max but for it to be used I would have had to generate almost equal number of non car images.

### Data Distribution

Made sure that almost equal number of car and non-car test images were available for training:

![distribution]

Initially a lot of yellow lane lines were being wrongly identified as cars, therefore I augmented the training data with some non-vehicle images generated from video frames. About 521 such images have been added. Following is a small sample of such images:

![augmented-not-cars]

### Histogram of Oriented Gradients (HOG)

#### Color Spaces Exploration

Explored, LUV, LAB, RGB, YUV and YCrCb color spaces.

Initially I checked out various color spaces visually to see, which ones were clearly indicating a car. For this I wrote [exploration.ipynb](exploration.ipynb) that would randomly pick up 5 car and 5 non-car samples, convert them to a particular color space and would display their individual channels on screen (Code block # 2). 

![cars-lab]
![notcars-lab]

#### Hog Display

Tried various HOG parameters by changing values manually and visually looking at, which values would clearly indicate cars. Some values of pix_per_cell, cell_per_block and orient were giving very good indications of car but feature size was also a big factor in deciding the final choice.

![car-hogs]
![notcar-hogs]
![combined-hogs]

In order to figure out, which one would work best for training, I wrote a block in [training.ipynb](training.ipynb) 7th block, where I've written code that will explorer multiple options of various color spaces, pix_per_cell, cell_per_block and orientation

```
training = []

all_params = [
    {
        'orient' : [12,11,9,8],
        'pix_per_cell' : [16, 8], 
        'cell_per_block' : [2, 4],
        'color_space' : ['YUV', 'YCrCb', 'LAB'],
        'hog_channel' : 'ALL',
        'spatial_feat' : True,
        'hist_feat' : True,
        'hog_feat' : True,
        'spatial_size' : (32,32),
        'hist_bins' : 32,
        'kernel' : 'linear',
        'C': 1,
        'gamma': 'auto'
    },
]

```
![various-options]

But I noticed that most of these were over fitting the training set. I tried running various combinations on the video file and all of them had some sort of false identifications and none of them was proving to be highly accurate. Therefore in the end I settled for a bigger orientation to allow for a smaller hog feature size.

Taking a hint from this post: [Good Tips from Reviewer](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903) I settled for the final parameters.

### Other Stuff Explored

*Laplacian Hog*

Also tried creating a Laplacian from RGB and then using HOG but that didn't prove to useful.

*Sobel X / Y*

Tried using Sobel X, Sobel Y or a combination of them but that actually decreased the overall fit on the training set

#### HOG Final Choice

In the end, I settled for the following parameters defined in [training.ipynb](training.ipynb) block labeled *Features Chosen*:

|Parameter|Value|
|-|-|
|pix_per_cell  |16  |
|cell_per_block | 2  |
|orient | 11  |
|spatial_feat | True  |
|hist_feat | True  |
|hog_feat | True |
|color_spaces | ['YUV']  |
|hog_channel | [0]| 
|spatial_size | (32,32)|
|hist_bins | 32|
|kernel | linear|
|C | 1|

Total feature size:  **4356**

#### Classifier Training

I've used a LinearSVC ([training](training.ipynb) Block Labelled: 'SVM Training') with the above parameters. For normalization I've used StandardScalar. The trained SVC, X_Scaler and the Parameters are pickled to a file for later processing in  [car-finder.py](car_finder.py)

# Learning

## Sliding Window Search

#### Window Scales

I settled on four different sizes based on detecting cars that are close by, then further up and then farthest. I have used my own functions for defining these window sizes [windows.py](windows.py).

In the following image the left hand side shows individual window sizes with no overlapping, just to give an idea of where the boxes search. On the right hand side it shows the same boxes with overlapping.

These are defined in [process.ipynb](process.ipynb) block labeled *Different Sliding Window Sizes*

|Window Name|Window Size|Overlapping|X Start / Stop | Y Start / Stop| # of windows
|-|-|-|-|-|-|
|Medium|176x176|0.4, 0|0, 1280|400,600|16|
|Small|112x112|0.3, 0.5|0, 1280|390,550|35|
|Smallest|90x90|0.4, 0.4|40, 1270|380,550|93|
|Tiny|90x90|0.4, 0.4|450,980|400,500|13|

Total: 157 windows

![window-shapes]

#### Car Detection

I've used YUV color space and the features include, 32x32 spatially binned image, color histogram and HOG of all channels.

There are a lot of false detections specially on the left hand side railing. To correct these, my foremost idea was to just include a lot of yellow lane lines and road railings as part of non-cars in the training set and retrain. This has helped somewhat but still a lot of times road railings are idenitifed as cars. Also oncoming traffic on the left hand side is also calssified as cars.

Classes Used:

**SingleFrameCarFinder** (line # 13 [car_finder.py](car_finder.py)): This class is reponsible for finding windows, adding heat map and generating heat_img for one frame only. I have written custom heat image generation code that overlaps heat map on top of an existing image for a better understanding.

**Vehicle** (line #173 [car_finder.py](car_finder.py)): This class is responsible for holding information for a particular identified car. I keep track of things like speed (in pixels per frame), frame_count, which frame did we see this car, bounding box and the number of times it has been as being classified

**VehicleIdentifier** (line # 261 [car_finder.py](car_finder.py)): This class processes each frame one by one. The following is the kind of logic that has been implemented:

+ Each frame, tell SingleFrameCarFinder (car_finder object) to identify, which windows say they have a car and to compute particular frame's heatmap
+ Append upto 10 heat maps in a deque
+ Sum up the 10 heat maps, each frame and then threshold it
+ From thresholded heatmap call Label to identify boxes
+ For each identified box, check if it belongs to a car that we already know off (self.cars) (Line #338)
+ If it does, good, tell the car to compute estimates of its speed, expand / contract its bounding box
+ If the box does not belong to a car, create a vehicle but add it to the set (maybe_cars)
+ Each frame, check maybe_cars to see if any box specified by the Label fits them
+ If for 5 frames, a maybe_car is still found to be there, classify it as a car and put it in self.cars. This is reason you would notice at the beginning of the video, threshold and Label have identified a car but the actual rectangle on the car takes a short while to appear. But this was ncessary since Label does a very bad job and some times it would show 3 cars rather than 2, so in order to get rid of these I am waiting for a few frames to make sure a car is identified
+ Each frame, check for cars that were there before but are no longer there any more (Line # 567)
+ For a car that has not been identified by the SVC, a new bounding box is created and it is re-searched by the SVC to make sure the car isn't in the frame any more (Line #587)

A low pass filter has been implemented that expands / moves the bounding box of each vehicle based on the last bounding box's center and the current bounding box's center (Line # 219) But I am not happy with the end result as the car is not completely enclosed

```
            new_width = old_width * (1 - self.alpha) + width * self.alpha
            new_height = old_height * (1 - self.alpha) + height * self.alpha

            self.center = self.center * (1 - self.center_alpha) + center * self.center_alpha
```

#### Threshold 

#### Test Images with Car Detection

The following image shows the different window sizes at work on different test images, where each column shows the particular detection for that scale of window.

![sliding-window-hascar]

#### False Detections

In order to inspect false detections in a frame / image, I wrote a block in [process.ipynb](process.ipynb) labelled *False Positives* where I show all detections in a given file/frame and a clsoeup (64x64) of the particular car. That paritcular close up can also also be saved to ./problems folder so that if the need be I can include that as part of training if needed.

![false-detection]

### Heatmap

Each frame being processed has four smaller images shown on top:

1. Left most shows windows that have been predicted to have a car and the heat map for the frame
2. Next one shows the heat map summed up without thresholding
3. 3rd one shows the heat map summed up and thresholded
4. Shows the classes identified as well as the number of cars detected

![sample-frame]

## Process Speedup

The process was so slow that I ended up cutting down to pix_per_cell = 16, cells_per_block = 2 and orient = 12 (which I later turned down to 11). 

To speed up the process, I wrote a multi threaded version, which did not turn out to be any faster. I found out later that python only used one CPU code so did not end up using that.


#### False Identifications

### Some sample frames:

False Identification got by passed by threshold for one or two frames:

![false-got-through.jpg]

False identification caught by thresholding:

![good-false]

Label not correctly telling us about 2 cars:

![bad_label]

# Discussion

#### Data Training

1) I am not at all happy with the false detections. Left hand side road railing is detected as a car a lot. Some times it is because of oncomming traffic and other times it is just plain old road. Altough thresholding catches that but still it requires better training

2) Personally, I think a deep convolution network would be a far better at prediction than HOG. Would like to transfer learning from ImageNet may be and extract features out of that and use those for SVM.

#### Boxes

1) I am not at all happy with the bounding algorithm used. Label, combines two blocks of cars, where as they have a clear center in the heat map.

![bad-label]

2) The low pass filter implemented for bounding box update is not that great. Would like to update this as well as other estimator with a kalman filter.

3) Towards the end of the video, the algorithm finds three cars as it keeps on thinking that the box for the black car has two cars in it. It is a very reduamentary algorithm and definitely needs to be worked upon for better car detection.

4) Thresholding: I just sum up the last 10 frames' heatmap. Some times when the oldest heatmap goes out of the deque, it so happens that it had a lot of numbers so all of a sudden there is a big change in the thresholded heatmap.

#### Faster Process

The process is still very slow at 0.5 secs / frame. Would like to increase this.

#### Z Space

There is no concept of depth in the code. If somehow I can tell, which car is behind, may be that can be used to better tell when an overlap occurs.

