# Vehicle Detection Project


[//]: # (Image References)
[distribution]: ./output_images/data-distribution.png
[augmented-not-cars]: ./output_images/augmented-not-cars.png
[cars-lab]: ./output_images/cars-lab.png
[notcars-lab]: ./output_images/notcars-lab.png
[car-hogs-8x4x9]: ./output_images/car-hogs-8x4x9.png
[car-hogs-8x4x12]: ./output_images/car-hogs-8x4x12.png
[car-hogs-16x4x12]: ./output_images/car-hogs-16x4x12.png
[notcar-hogs-8x4x9]: ./output_images/notcar-hogs-8x4x9.png
[notcar-hogs-8x4x12]:./output_images/notcar-hogs-8x4x12.png
[notcar-hogs-16x4x12]:./output_images/notcar-hogs-16x4x12.png
[window-shapes]:./output_images/window_shapes.png
[sliding-window-hascar]:./output_images/sliding_window_hascar.png
[false-detection]:./output_images/false_detection.png
[heatmap]:./output_images/heatmap.jpg
[sample-missframe]:./output_images/0595.jpg

## Project Video 

Project video is available at [./project-video-cars.mp4](./project-video-cars.mp4)

## Training SVM

Overall solution's training part has been coded in jupyter notebook and the processing pipeline as been coded as python files:

[VehicleDetection-Training](training.ipynb)  
[pipeline.py](pipeline.py)  
[pipeline_funcs.py](pipeline_functions.py)

### Data Used

Mostly used GTI and KITTI car / not-car images that were given as part of [vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

During testing a lot of times the yellow lane line was being misclassified as a car, therefore I generated a number of smaller 64x64 images from the first 100 frames of the video, which had no car visible and used them as part of non-vehicles set.

I tried using the udacity's labeled data set [Udacity Labeled Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) but was surprised to see that it had major issues. A lot of frames' bounding box is not correct. Hence in the end I did not use that. [extract_udacity.ipynb](extract_udacity.ipynb)

### Data Distribution

Made sure that almost equal number of car and non-car test images were available for training:

![distribution]

Initially a lot of yellow lane lines were being wrongly identified as cars, therefore I augmented the training data with some non-vehicle images generated from video frames. About 521 such images have been added. Following is a small sample of such images:

![augmented-not-cars]

### Histogram of Oriented Gradients (HOG)

#### Color Spaces Exploration

Explored, LUV, LAB, RGB and YCrCb color spaces.

To check each color space, wrote code that would randomly pick up 5 car and 5 non-car samples, convert them to a particular color space and would display their individual channels on screen (Code block # 2 of [training]). 

![cars-lab]
![notcars-lab]

#### Hog Display

Tried various HOG parameters by changing values manually but later on wrote a jupyter notebook (TestSVMParams) in which the following parameters were defined in a dictionary, each one was run against the same Train / Test split and in the end the best one was chosen.


```
params = [
    {
        'pix_per_cell' : 16,
        'cell_per_block' : 4,
        'orient': 12,
    },
    {
        'pix_per_cell' : 8,
        'cell_per_block' : 4,
        'orient': 9,
    },
    {
        'pix_per_cell' : 8,
        'cell_per_block' : 4,
        'orient': 12,
    },
    {
        'pix_per_cell' : 16,
        'cell_per_block' : 2,
        'orient': 9,
    },
    {
        'pix_per_cell' : 16,
        'cell_per_block' : 2,
        'orient': 16,
    },
    {
        'pix_per_cell' : 17,
        'cell_per_block' : 2,
        'orient': 16,
    },
]

```

In order to make sure that each parameter was being run against the same set of train/test split, all filenames were combined into an array and the indices of that array were split between training / testing sets. 

```
all_filenames = np.hstack((cars, notcars))

X_indices = np.arange(len(all_filenames))
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

rand_state = np.random.randint(0, 100)
X_train_indices, X_test_indices, y_train, y_test = train_test_split(X_indices, y, test_size=0.2, random_state=rand_state)

```

**Some HOG visual displays:**

*Pix/Cell: 8, Cells/Block: 4, Orientation: 9, Feature Size/Channel: **3600***

![car-hogs-8x4x9]
![notcar-hogs-8x4x9]

*Pix/Cell: 8, Cells/Block: 4, Orientation: 12, Feature Size/Channel: **4800***

![car-hogs-8x4x12]
![notcar-hogs-8x4x12]


*Pix/Cell: 16, Cells/Block: 4, Orientation: 12, Feature Size/Channel: **192***

![car-hogs-16x4x12]
![notcar-hogs-16x4x12]

### Other Stuff Explored

*Laplacian Hog*

Also tried creating a Laplacian from RGB and then using HOG but that didn't prove to useful.

*Sobel X / Y*

Tried using Sobel X, Sobel Y or a combination of them but that actually decreased the overall fit on the training set

#### HOG Final Choice

Ran LinearSVM for each of the above given parameters for three color spaces LUV, LAB and YCrCb. In the end, I settled for the following parameters defined in [process.ipynb](process.ipynb) block labeled *Features Chosen*:

|Parameter|Value|
|-|-|
|pix_per_cell  |8  |
|cell_per_block | 4  |
|orient | 12  |
|spatial_feat | True  |
|hist_feat | True  |
|hog_feat | True |
|color_spaces | ['LAB']  |
|hog_channel | [0]| 
|spatial_size | (32,32)|
|hist_bins | 32|

Total feature size:  **7968**

#### Classifier Training

I've used a LinearSVC ([training](training.ipynb) Block Labelled: 'SVM Training') with the above parameters. For normalization I've used StandardScalar, but I do not include the test set in normalizing. Only the training set is used for normalizing and then the test is transformed using the parameters learned from training.

```
car_features = extract_features(cars, color_space=color_space, orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat = spatial_feat,
                               spatial_size = spatial_size,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat = spatial_feat,
                               spatial_size = spatial_size,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

X_train_scaler = StandardScaler().fit(X_train)
X_train = X_train_scaler.transform(X_train)
X_test = X_train_scaler.transform(X_test)

# Learning

svc = LinearSVC()
svc.fit(X_train, y_train)

print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

```

## Sliding Window Search

#### Window Scales

I settled on three different scales based on detecting cars that are close by, then furher up and then farthest. In the following image the left hand side shows individual window sizes with no overlapping, just to give an idea of where the boxes search. On the right hand side it shows the same boxes with overlapping.

These are defined in [process.ipynb](process.ipynb) block labeled *Different Sliding Window Sizes*

|Window Name|Window Size|Overlapping|X Start / Stop | Y Start / Stop|
|-|-|-|-|-|
|Medium|175x160|0.4, 0|0, img.shape[1]|400,600|
|Small|160x120|0.3, 0.5|0, img.shape[1]|390,550|
|Smallest|80x70|0.2, 0.2|100, img.shape[1] - 100|400,480|

![window-shapes]

#### Car Detection

I've used LAB color space and the features include, 32x32 spatially binned image, color histogram and HOG of all channels.

There are a lot of false detections specially on the left hand side railing. To correct these, my foremost idea was to just include a lot of yellow lane lines and road railings as part of non-cars in the training set and retrain. This has helped somewhat but still a lot of times road railings are idenitifed as cars.

A class has been written *VehicleIdentifier* [pipeline.py](pipeline.py) Line #: 97, that takes the image and processes it. Each frame's heat map is summed up and thresholded. After thresholding another class *Vehicle* (Line # 29)  is created for all vehicles identified.

A low pass filter has been implemented that expands / moves the bounding box of each vehicle based on the last bounding box's center and the current bounding box's center (Line # 76) But I am not happy with the end result as the car is not completely enclosed

```
            new_width = old_width * (1 - self.alpha) + width * self.alpha
            new_height = old_height * (1 - self.alpha) + height * self.alpha

            self.center = self.center * (1 - self.center_alpha) + center * self.center_alpha
```

Sometimes the detection goes off a little and the heat map marks the visible car as not ok any more. For such frames, have implemented the logic that each identified car has a chance of 10 miss frames during, which I check if there was a bounding box in the same region, regardless of whether the heatmap was hot enough or not, the bounding box is still considered as good. Line # 213 of pipeline.py has this code.

![sample-missframe]
#### Test Images with Car Detection

The following image shows the different window sizes at work on different test images, where each column shows the particular detection for that scale of window.

![sliding-window-hascar]

#### False Detections

In order to inspect false detections in a frame / image, I wrote a block in [process.ipynb](process.ipynb) labelled *False Positives* where I show all detections in a given file/frame and a clsoeup (64x64) of the particular car. That paritcular close up is also saved to ./problems folder so that if the need be I can include that as part of training if needed.

![false-detection]

### Heatmap

Each frame being processed has four smaller images shown on top:

1. Left most shows the idenitifications within this frame
2. Next one shows the heat map for the current frame
3. 3rd one shows the heat map summed up and thresholded
4. Shows the classes identified as well as the number of cars detected

![heatmap]

## Process Speedup

The process was so slow that I ended up cutting down to pix_per_cell = 16, cells_per_block = 2 and orient = 12. 

Then I wrote code that creates 20 threads and about 124 boxes are distributed amongst the 20 threads to look for cars within the thread. This is coded in [pipeline_funcs.py](pipeline_funcs.py) line # 38 *class CarDetector*.


## Video Implementation

Project video is available at [./project-video-cars.mp4](./project-video-cars.mp4)

#### False Identifications


### Here are six frames and their corresponding heatmaps:


### Discussion

#### Data Training

1) I am not at all happy with the false detections. Left hand side road railing is detected as a car a lot
2) Personally, I think a deep convolution network would be a far better at prediction than HOG. Would like to transfer learning from ImageNet may be and extract features out of that and use those for SVM.

#### Boxes

1) I am not at all happy with the bounding algorithm used. Right now it just considers all of the boxes that are in the heat map and the bigger box towards the bottom frame just makes the box more elongated. Maybe would later on implement a dynamic box that expands (or contracts) to come up with a best fit window.

2) The low pass filter implemented for bounding box update is not that great. Would like to update this with a kalman filter.

3) Had implemented a rudementary algorithm to detect the speed of the vehicle so that in case if it is not detected I can at least try to estimate and then resample that particular area may be. But the routine that I had written (Line # 102) ruins the estimate.

#### Faster Process

The process is still very slow at 0.5 secs / frame. Would like to increase this.

#### Hog Subsampling

In order to speed up the process, tried implementing Hog Subsampling using a different approach than the one proposed in the class lectures. 

Created the hog of the 1280x700 image, extracted hog blocks for the whole of the sliding window being searched for, which would result in much more number of pixels than the 64x64 image that we trained on therefore after sampling reshaped the pixels of the block so that I have the same number of rows as the original 64x64's pixel and then took the mean of each row. But in the end didnot use this since as is I was getting bad detections. The code has been written in 
