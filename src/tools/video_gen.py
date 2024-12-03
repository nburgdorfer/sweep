import os
import argparse
from tqdm import tqdm
import sys
import cv2
import numpy as np
import imageio.v2 as iio

TOOLS_ROOT=os.path.abspath(os.path.dirname(__file__))
SRC_ROOT=os.path.abspath(os.path.dirname(TOOLS_ROOT))
TOP_ROOT=os.path.abspath(os.path.dirname(SRC_ROOT))
sys.path.append(TOP_ROOT)

from src import config

def generate_gif_1(vid_file, image_paths):
    image_files = os.listdir(image_paths[0])
    image_files.sort()
    with iio.get_writer(vid_file, format="GIF-PIL", mode="I", fps=fps, loop=0) as writer:
        for imf in tqdm(image_files, desc="Generating GIF", unit="frames"):
            image = cv2.imread(os.path.join(image_paths[0], imf))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)


def generate_video_1(vid_file, image_paths):
    image_files = os.listdir(image_paths[0])
    image_files.sort()
    with iio.get_writer(vid_file, format="FFMPEG", mode="I", fps=fps) as writer:
        for imf in tqdm(image_files, desc="Generating Video", unit="frames"):
            image = cv2.imread(os.path.join(image_paths[0], imf))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)


def generate_gif_2(vid_file, image_paths):
    images_path_1 = image_paths[0]
    images_path_2 = image_paths[1]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    with iio.get_writer(vid_file, format="GIF-PIL", mode="I", fps=fps, loop=0) as writer:
        for i in tqdm(range(num_images), desc="Generating GIF", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            frame = np.concatenate((image_1,image_2), axis=1)
            writer.append_data(frame)

def generate_video_2(vid_file, image_paths, fps):
    images_path_1 = image_paths[0]
    images_path_2 = image_paths[1]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    with iio.get_writer(vid_file, format="FFMPEG", mode="I", fps=fps) as writer:
        for i in tqdm(range(num_images), desc="Generating Video", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            
            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            frame = np.concatenate((image_1,image_2), axis=1)
            writer.append_data(frame)

def generate_gif_3(vid_file, image_paths):
    images_path_1 = image_paths[0]
    images_path_2 = image_paths[1]
    images_path_3 = image_paths[2]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    image_files_3 = os.listdir(images_path_3)
    image_files_3.sort()

    with iio.get_writer(vid_file, format="GIF-PIL", mode="I", fps=fps, loop=0) as writer:
        for i in tqdm(range(num_images), desc="Generating GIF", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_3 = cv2.imread(os.path.join(images_path_3, image_files_3[i]))
            image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

            frame = np.concatenate((image_1,image_2, image_3), axis=1)
            writer.append_data(frame)

def generate_video_3(vid_file, image_path, fps):
    images_path_1 = image_path[0]
    images_path_2 = image_path[1]
    images_path_3 = image_path[2]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    image_files_3 = os.listdir(images_path_3)
    image_files_3.sort()

    with iio.get_writer(vid_file, format="FFMPEG", mode="I", fps=fps) as writer:
        for i in tqdm(range(num_images), desc="Generating Video", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_3 = cv2.imread(os.path.join(images_path_3, image_files_3[i]))
            image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

            frame = np.concatenate((image_1,image_2, image_3), axis=1)
            writer.append_data(frame)

def generate_gif_4(vid_file, images_paths):
    images_path_1 = image_paths[0]
    images_path_2 = image_paths[1]
    images_path_3 = image_paths[2]
    images_path_4 = image_paths[3]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    image_files_3 = os.listdir(images_path_3)
    image_files_3.sort()

    image_files_4 = os.listdir(images_path_4)
    image_files_4.sort()

    with iio.get_writer(vid_file, format="GIF-PIL", mode="I", fps=fps, loop=0) as writer:
        for i in tqdm(range(num_images), desc="Generating GIF", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_3 = cv2.imread(os.path.join(images_path_3, image_files_3[i]))
            image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

            image_4 = cv2.imread(os.path.join(images_path_4, image_files_4[i]))
            image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB)

            frame_top = np.concatenate((image_1,image_2), axis=1)
            frame_bottom = np.concatenate((image_3,image_4), axis=1)
            frame = np.concatenate((frame_top,frame_bottom), axis=0)
            
            writer.append_data(frame)

def generate_video_4(vid_file, image_paths, fps):
    images_path_1 = image_paths[0]
    images_path_2 = image_paths[1]
    images_path_3 = image_paths[2]
    images_path_4 = image_paths[3]

    image_files_1 = os.listdir(images_path_1)
    image_files_1.sort()
    num_images = len(image_files_1)
    
    image_files_2 = os.listdir(images_path_2)
    image_files_2.sort()

    image_files_3 = os.listdir(images_path_3)
    image_files_3.sort()

    image_files_4 = os.listdir(images_path_4)
    image_files_4.sort()

    with iio.get_writer(vid_file, format="FFMPEG", mode="I", fps=fps) as writer:
        for i in tqdm(range(num_images), desc="Generating Video", unit="frames"):
            image_1 = cv2.imread(os.path.join(images_path_1, image_files_2[i]))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

            image_2 = cv2.imread(os.path.join(images_path_2, image_files_2[i]))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_3 = cv2.imread(os.path.join(images_path_3, image_files_3[i]))
            image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

            image_4 = cv2.imread(os.path.join(images_path_4, image_files_4[i]))
            image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB)

            frame_top = np.concatenate((image_1,image_2), axis=1)
            frame_bottom = np.concatenate((image_3,image_4), axis=1)
            frame = np.concatenate((frame_top,frame_bottom), axis=0)

            writer.append_data(frame)

def video_gen(output_path, image_paths, vid_file, fps):
    vid_file = os.path.join(output_path, vid_file)
    num_windows = len(image_paths)
    image_paths = [os.path.join(output_path, ip) for ip in image_paths]

    if (num_windows == 4):
        if (vid_file[-3:]==".gif"):
            generate_gif_4(vid_file, image_paths[:4], fps)
        else:
            generate_video_4(vid_file, image_paths[:4], fps)
            
    elif (num_windows == 3):
        if (vid_file[-3:]==".gif"):
            generate_gif_3(vid_file, image_paths[:3], fps)
        else:
            generate_video_3(vid_file, image_paths[:3], fps)

    elif (num_windows == 2):
        if (vid_file[-3:]==".gif"):
            generate_gif_2(vid_file, image_paths[:2], fps)
        else:
            generate_video_2(vid_file, image_paths[:2], fps)

    elif (num_windows == 1):
        if (vid_file[-3:]==".gif"):
            generate_gif_1(vid_file, image_paths[:1], fps)
        else:
            generate_video_1(vid_file, image_paths[:1], fps)

    else:
        print("Invalid number of image frames to stitch into a video.")
        sys.exit()
