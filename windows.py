import time
import cv2
import numpy as np

def get_windows(img, x_start_stop=[None, None], y_start_stop=[None,None],
              window_size=(64,64), offset_factor=(1,1), no_of_windows = None, draw_color = None):

    x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
    x_end = x_start_stop[1] if x_start_stop[1] is not None else img.shape[1]
    y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
    y_end = y_start_stop[1] if y_start_stop[1] is not None else img.shape[0]

    windows = []

    # in case we want a negative offset e.g. want to go from right most side of the
    # image to the left most then we generate reversed x_pts
    if offset_factor[0] > 0:
        x_pts = np.arange(x_start, x_end, window_size[0] * offset_factor[0]).astype(np.int_)
    else:
        x_pts = np.arange(x_end - window_size[0], x_start, window_size[0] * offset_factor[0]).astype(np.int_)
        
    if offset_factor[1] > 0:
        y_pts = np.arange(y_start, y_end, window_size[1] * offset_factor[1]).astype(np.int_)
    elif offset_factor[1] == 0:
        y_pts = [y_start]
    else:
        y_pts = np.arange(y_end - window_size[1], y_start, window_size[1] * offset_factor[1]).astype(np.int_)
    
    no_of_windows = no_of_windows if no_of_windows is not None else len(x_pts) * len(y_pts)
    
    for y in y_pts:
        for x in x_pts:
            x2 = x + window_size[0]
            y2 = y + window_size[1]
            
            if x2 > x_end or y2 > y_end:
                continue

            windows.append(((x,y),(x2,y2)))
            if len(windows) >= no_of_windows:
                break
        if len(windows) >= no_of_windows:
            break
 
    if draw_color is not None:
        for window in windows:
            cv2.rectangle(img, window_size[0], window_size[1], draw_color, 4)

    return windows

def medium_window(img, draw_color = None, no_of_windows = None, offset = (-0.4, 0), size_only = False):
    window_size=(11*16, 11*16)

    if size_only:
        return window_size

    x_start = 0
    x_stop = img.shape[1]
    y_start = 400
    y_stop = 600
    
    return get_windows(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          window_size=window_size, 
          draw_color = draw_color, 
          no_of_windows = no_of_windows,
          offset_factor = offset)
    
def small_window(img, draw_color = None, no_of_windows = None, offset = (-0.3, 0.5), size_only = False):
    window_size=(7*16, 7*16)

    if size_only:
        return window_size

    x_start = 0
    x_stop = img.shape[1]
    y_start = 390
    y_stop = 550

    if size_only:
        return window_size

    return get_windows(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          window_size=window_size, 
          draw_color=draw_color, 
          no_of_windows = no_of_windows,
          offset_factor = offset)


def smallest_window(img, draw_color = None, no_of_windows = None, offset = (-0.4, 0.4), size_only = False):
    window_size=(90,90)

    if size_only:
        return window_size

    #x_start = 600
    # window = (80,70)
    x_start = 40
    x_stop = img.shape[1] - 40
    y_start = 380
    y_stop = 550

    return get_windows(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          window_size=window_size, 
          draw_color = draw_color, 
          no_of_windows = no_of_windows,
          offset_factor=offset)

def tiny_window(img, draw_color = None, no_of_windows = None, offset = (-0.4, 0.5), size_only = False):
    window_size=(86,86)

    if size_only:
        return window_size

    x_start = 450
    x_stop = img.shape[1] - 300
    y_start = 400
    y_stop = 500

    return get_windows(img, 
          x_start_stop = (x_start, x_stop), 
          y_start_stop = (y_start, y_stop),
          window_size = window_size,
          draw_color = draw_color, 
          no_of_windows = no_of_windows,
          offset_factor=offset)

def get_all_window_sizes(img):
    windows = []
    #windows.extend(big_window(img))
    windows.append(medium_window(img, size_only = True))
    windows.append(small_window(img, size_only = True))
    windows.append(smallest_window(img, size_only = True))
    windows.append(tiny_window(img, size_only = True))
    return windows

def get_all_windows(img):
    windows = []
    #windows.extend(big_window(img))
    windows.extend(medium_window(img))
    windows.extend(small_window(img))
    windows.extend(smallest_window(img))
    windows.extend(tiny_window(img))
    return windows


if __name__ == '__main__':
    print(len(get_all_windows(np.zeros(shape=(720,1280,3)))))
    print(len(get_all_window_sizes(np.zeros(shape=(720,1280,3)))))
