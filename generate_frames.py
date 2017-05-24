import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from glob import glob
import shutil

def mark_all_frames(video_filename):
    output = os.path.splitext(video_filename)
    output_folder = "{}-frames".format(output[0])
    
    if os.path.exists(output_folder):
        print('removing folder', output_folder)
        shutil.rmtree(output_folder)
    
    os.mkdir(output_folder)
    print('Folder created', output_folder)
    
    clip = VideoFileClip(video_filename)

    for i, image in enumerate(clip.iter_frames()):
        mpimg.imsave("{}/{:04d}.jpg".format(output_folder, i), image, format="jpeg")
        print("{} generated".format(i), end='\r')
        
    print("done")

mark_all_frames('project_video-29-39.mp4')