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
                                visualise=vis, feature_vector=feature_vec,
                                block_norm = 'L2-Hys')
    #return features, hog_image
    #return features
    return res

def get_laplacian_hog(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    img = cv2.GaussianBlur(img, (3,3), 0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    hog_features = []    
    for channel in range(img.shape[2]):
        channel_features = hog(laplacian[:,:,channel], orientations=orient, 
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), 
                                transform_sqrt=False, 
                                visualise=vis, 
                                feature_vector=feature_vec,
                                block_norm='L2-Hys')
        hog_features.append(channel_features)

    #return features, hog_image
    #return features
    return np.ravel(hog_features)

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_feature_image(feature_image, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat = True, 
                        hist_feat = True, 
                        hog_feat = True,
                        laplacian_feat = True):

    file_features = []

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    
    if laplacian_feat == True:
        lap_hog = get_laplacian_hog(feature_image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        file_features.append(lap_hog)

    return np.concatenate(file_features)
    

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, laplacian_feat = True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        feature_image = load_image(file, color_space)
        file_features = extract_feature_image(feature_image, spatial_size=spatial_size,
                            hist_bins = hist_bins, orient=orient, pix_per_cell = pix_per_cell,
                            cell_per_block=cell_per_block, hog_channel = hog_channel,
                            spatial_feat = spatial_feat, hist_feat = hist_feat, hog_feat = hog_feat, laplacian_feat = laplacian_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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


def load_image(file, color_space):
    # Read in each one by one
    img = cv2.imread(file)
    space = eval('cv2.COLOR_BGR2' + color_space)
    return cv2.cvtColor(img, space)

def laplacian_features(channel):
    img = cv2.GaussianBlur(channel, (3,3), 0)
    img_32 = cv2.resize(img, (32,32))
    laplacian_32 = cv2.Laplacian(img_32, cv2.CV_64F)
    return laplacian_32.ravel()

def get_boxes(img, x_start_stop=[None, None], y_start_stop=[None,None],
              box=(64,64), offset_factor=(1,1), no_of_boxes = None, draw_color = None):

    x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
    x_end = x_start_stop[1] if x_start_stop[1] is not None else img.shape[1]
    y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
    y_end = y_start_stop[1] if y_start_stop[1] is not None else img.shape[0]

    boxes = []

    # in case we want a negative offset e.g. want to go from right most side of the
    # image to the left most then we generate reversed x_pts
    if offset_factor[0] > 0:
        x_pts = np.arange(x_start, x_end, box[0] * offset_factor[0]).astype(np.int_)
    else:
        x_pts = np.arange(x_end - box[0], x_start, box[0] * offset_factor[0]).astype(np.int_)
        
    if offset_factor[1] > 0:
        y_pts = np.arange(y_start, y_end, box[1] * offset_factor[1]).astype(np.int_)
    elif offset_factor[1] == 0:
        y_pts = [y_start]
    else:
        y_pts = np.arange(y_end - box[1], y_start, box[1] * offset_factor[1]).astype(np.int_)
    
    no_of_boxes = no_of_boxes if no_of_boxes is not None else len(x_pts) * len(y_pts)
    
    for y in y_pts:
        for x in x_pts:
            x2 = x + box[0]
            y2 = y + box[1]
            
            if x2 > x_end or y2 > y_end:
                continue

            boxes.append(((x,y),(x2,y2)))
            if len(boxes) >= no_of_boxes:
                break
        if len(boxes) >= no_of_boxes:
            break
 
    if draw_color is not None:
        for box in boxes:
            cv2.rectangle(img, box[0], box[1], draw_color, 4)

    return boxes
