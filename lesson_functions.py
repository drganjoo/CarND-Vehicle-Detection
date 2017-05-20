import numpy as np
import cv2
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    res = hog(img, orientations=orient, 
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), 
                                transform_sqrt=True, 
                                visualise=vis, 
                                feature_vector=feature_vec,
                                block_norm='L1-sqrt')
    return res

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # if (np.max(img[:,:,0]) <= 1) and (bins_range[1] > 1):
    #     print('Looks like you are reading PNG with 0..1 but calling bins_range with 0..256')

    # Compute the histogram of the color channels separately
    channel_hist = []
    if len(img.shape) > 2:
        for i in range(img.shape[-1]):
            channel_hist.append(np.histogram(img[:,:,i], bins=nbins, range=bins_range)[0])
    else:
        channel_hist.append(np.histogram(img, bins=nbins, range=bins_range)[0])

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(channel_hist)
    return hist_features

def load_image(file, color_space):
    # Read in each one by one
    img = cv2.imread(file)
    # apply color conversion if other than 'RGB'
    space = eval('cv2.COLOR_BGR2' + color_space)
    return cv2.cvtColor(img, space)

def laplacian_features(channel):
    img = cv2.GaussianBlur(channel, (3,3), 0)
    img_32 = cv2.resize(img, (32,32))
    laplacian_32 = cv2.Laplacian(img_32, cv2.CV_64F)
    return laplacian_32.ravel()

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel = [0],
                        spatial_feat=True, 
                        hist_feat=False, 
                        laplacian_feat = True,
                        hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []

        feature_image = load_image(file, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if laplacian_feat == True:
            lap_features = laplacian_features(feature_image[:,:,0])
            file_features.append(lap_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = []
            for channel in hog_channel:
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)

            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), 
                    xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.abs(sobel)
    scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))

    mask = (scaled >= thresh[0]) & (scaled <= thresh[1])

    grad_binary = np.zeros_like(channel)
    grad_binary[mask] = 1
    return grad_binary


def mag_thresh(channel, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled = np.uint8(255.0 * mag / np.max(mag))

    mask = (scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])

    mag_binary = np.zeros_like(channel)
    mag_binary[mask] = 1
    return mag_binary

def dir_threshold(channel, sobel_kernel=3, thresh_deg=(0, 90)):
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    dir_sobel = np.arctan2(np.abs(sobely), np.abs(sobelx))

    thresh_rad = [thresh_deg[0] * np.pi / 180.0, thresh_deg[1] * np.pi / 180.0]
    mask = (dir_sobel >= thresh_rad[0]) & (dir_sobel <= thresh_rad[1])

    dir_binary = np.zeros_like(channel)
    dir_binary[mask] = 1
    return dir_binary

