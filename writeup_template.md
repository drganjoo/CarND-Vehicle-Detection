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
[heatmap]:./output_images/heatmap.png

## Training SVM

Overall solution has been coded in two jupyter notebooks:

[VehicleDetection-Training](training.ipynb)  
[VehicleDetection-Process](process.ipynb)

### Data Used

Mostly used GTI and KITTI car / not-car images that were given as part of [vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles.zip](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

During testing a lot of times the yellow lane line was being misclassified as a car, therefore I generated a number of smaller 64x64 images from the first 100 frames of the video, which had no car visible and used them as part of non-vehicles set.

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

I've used LAB color space and the features include, 32x32 spatially binned image, histogram of colors for all three channels and HOG of the L Channel.

There are some false detections, mostly of yellow lane lines. To correct these, my foremost idea was to just include a lot of yellow lane lines as non-cars in the training set and retrain. This has helped a lot but still there are stages where a yellow lane is detected as car.

#### Test Images with Car Detection

The following image shows the three different window sizes at work on different test images, where each column shows the particular detection for that scale of window.

![sliding-window-hascar]

#### False Detections

In order to inspect false detections in a frame / image, I wrote a block in [process.ipynb](process.ipynb) labelled *False Positives* where I show all detections in a given file/frame and a clsoeup (64x64) of the particular car. That paritcular close up is also saved to ./problems folder and then I can include that as part of training if needed.

![false-detection]

### Heatmap Test

To figure out what the heatmap looks like for a given image, a block has been written in [process.ipynb](process.ipynb) labelled *'Heatmap Test'*, which clearly shows which windows found a car, what the heatmap was for the particular detection, what "label" detected as a car and finally the bounded box.

![heatmap]


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

