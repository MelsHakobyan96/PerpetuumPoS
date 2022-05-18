from traceback import print_list
import cv2
import os
import sys
import torch

import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from utils import *


def save_images_as_video(frame_list, video_path, extension=None, frame_rate=20, size=(1164, 874)):
    """
        Given the frames it generates a video of the specified extension.

        Args:
            frame_list: list of video images,
            video_path: the path to save the video,
            extension: the desired extension of the video,
            frame_rate: the frame rate of the video,
            size: the (width, height) of a frame of the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if extension is not None:
        video_path += extension
    writer = cv2.VideoWriter(video_path, fourcc, frame_rate, size)

    for frame in frame_list:
        writer.write(frame)

    writer.release()


def save_frames_as_video(frame_list, video_path, frame_rate=20):
    """
        Given the frames it generates a video.

        Args:
            frame_list: list of video images,
            video_path: the path to save the video,
            frame_rate: the frame rate of the video,
    """
    clip = ImageSequenceClip(frame_list, fps=frame_rate)
    clip.write_videofile(video_path, codec='mpeg4', logger=None)


def get_frames(video_path, chunk_size=16, dataset=False):
    """
        Genrator yeilding frames of a video by spliting it into batch_size chunks.

        Args:
            video_path: the path to save the video,
            chunk_size: number of frames in each chunk,
            dataset: True if it is used for pytorch Dataset class.
    """
    frame_list = []
    frame_iter = 0

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    frame_list.append(frame)
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)

        if len(frame_list) == chunk_size and not dataset:
            yield frame_list, frame_iter

            frame_iter += 1
            frame_list = []

    yield np.stack(frame_list)


def split_video_into_chunks(dir_path, chunk_size):
    """
        Splits the video into smaller video of the specified batch size.

        Args:
            dir_path: the path to the dirctory where the data is,
            chunk_size: number of frames you want the smaller videos to have.
    """
    video_path = os.path.join(dir_path, 'video.hevc')
    video_chunk_folder_path = os.path.join(dir_path, 'videos')

    if not os.path.exists(video_chunk_folder_path):
        os.mkdir(video_chunk_folder_path)

    last_index = 0
    for frames, idx in get_frames(video_path, dataset=False):
        if len(frames) < chunk_size:
            last_index = idx
            break

        chunk_path = os.path.join(video_chunk_folder_path, f'chunk_{idx}')
        save_images_as_video(frames, chunk_path)

    return last_index


def process_directory_paths(dir_paths):
    """
        Adding processed video and sensor folder names into csv.
        Args:
            dir_paths: list of directory path,
            dir_count: number of directories in each folder.
    """
    paths = []
    for path in tqdm(dir_paths, total=len(dir_paths)):
        files_count = len(os.listdir(os.path.join(path, '160_320_videos')))

        path = '/'.join(path.split('/')[-3:])
        paths.extend([(os.path.join(path, f'160_320_videos/chunk_{str(i)}_160_320.avi'),
                       os.path.join(path, f'sensors/chunk_{str(i)}.csv'))
                      for i in range(files_count)])
    return paths


def process_and_save_paths_as_csv(dir_paths, csv_path, column_names):
    """
        Saving list as a csv file.

        Args:
            dir_paths: list of directory path,
            csv_path: path to save the csv file,
            column_names: names of the columns of the DataFrame.
    """
    paths = process_directory_paths(dir_paths)
    df = pd.DataFrame(paths, columns=column_names)
    df.to_csv(csv_path, index=False)


def train_valid_test_split(dir_list, dir_path, test_size=0.2):
    """
        Splitting the data into train, valid and test chunks.

        Args:
            dir_list: list of dirctories,
            test_size: the ratio of the testing data.
    """
    train_data, testing_data = train_test_split(
        dir_list, test_size=test_size, random_state=1)

    valid_data, test_data = train_test_split(
        testing_data, test_size=0.5, random_state=1)

    dir_path = '/'.join(dir_path.split('/')[:-1])
    dir_path = os.path.join(dir_path, 'paths')

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    column_names = ['video_path', 'sensor_path']
    process_and_save_paths_as_csv(
        train_data, os.path.join(dir_path, 'train.csv'), column_names)
    process_and_save_paths_as_csv(
        valid_data, os.path.join(dir_path, 'valid.csv'), column_names)
    process_and_save_paths_as_csv(
        test_data, os.path.join(dir_path, 'test.csv'), column_names)


def list_of_directories(parent_dir_path):
    """
        A function that adds the folders into one big one.

        Args:
            parent_dir_path: the path to the parent dirctory.
    """
    dir_paths = []

    for dir in os.listdir(parent_dir_path):
        # running along the child directories
        for child_dir in os.listdir(os.path.join(parent_dir_path, dir)):
            child_dir_path = os.path.join(parent_dir_path, dir, child_dir)
            dir_paths.append(child_dir_path)

    return dir_paths


def split_sensor_data(dir_path, batch_size, video_count):
    """
        Splits the sensor data into specified batch size chunks.

        Args:
            dir_path: the path to the dirctory where the data is,
            batch_size: number of sensors you want chunks to have.
    """
    sensor_path = os.path.join(dir_path, 'sensors.csv')
    sensor_chunk_folder_path = os.path.join(dir_path, 'sensors')

    if not os.path.exists(sensor_chunk_folder_path):
        os.mkdir(sensor_chunk_folder_path)

    sensor_df = pd.read_csv(sensor_path)

    for i, idx in enumerate(range(0, len(sensor_df), batch_size)):
        sensor_chunk = sensor_df.iloc[idx: idx + batch_size, 1:]

        if len(sensor_chunk) < batch_size or idx == video_count - 1:
            break

        sensor_chunk_path = os.path.join(
            sensor_chunk_folder_path, f'chunk_{i}.csv')

        sensor_chunk.to_csv(sensor_chunk_path)


def generate_chunks(dir_list, batch_size=16):
    """
        A function genrating small videos and  the corresponding sensors data
        in one go while tracking the exeqution time.

        Args:
            dir_list: list of dirctories,
            batch_size: number of frames you want the smaller videos to have.
    """
    start = datetime.now()
    for dir in tqdm(dir_list, total=len(dir_list), desc='Chunk number: '):
        video_count = split_video_into_chunks(dir, batch_size)
        split_sensor_data(dir, batch_size, video_count)

    print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start))


def resize_frames(frames, size):
    """
        Given the frames resizes it to the specified size and returns a numpy array.

        Args:
            frames: numpy array of video images,
            size: the (width, height) of a frame of the video.
    """
    transform = transforms.Compose([
        transforms.Resize(size),
    ])

    frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
    frames = transform(frames)
    frames = frames.permute(0, 2, 3, 1)

    return frames.numpy()


def color_correct(frame):
    """
        Correcting the colors of the frame.

        Args:
            frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_and_save_images(dir_list, size=(160, 320)):
    """
        Resizes all the videos of the given directories to specified size.

        Args:
            dir_list: paths of directories,
            size: the (width, height) of a frame of the video.
    """
    for path in tqdm(dir_list, total=len(dir_list)):
        current_path = os.path.join(path, 'videos')
        new_path = os.path.join(path, f'{size[0]}_{size[1]}_videos')

        files = sorted(os.listdir(current_path))

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        for file_name in files:
            file_path = os.path.join(current_path, file_name)

            frames = next(get_frames(file_path, dataset=True))
            frames = resize_frames(frames, size)
            frames = [color_correct(arr.squeeze(0))
                      for arr in np.array_split(frames, 8)]

            new_file_name = file_name.split(
                '.')[0] + f'_{size[0]}_{size[1]}.avi'
            new_file_path = os.path.join(new_path, new_file_name)

            save_frames_as_video(frames, new_file_path)


def sensor_processing(sensors):
    """
        Reads and processes the sensor data taken from csv.

        Args:
            sensors: numpy array of sensor data.
    """
    sensors = sensors.iloc[:, -2:].values
    speed_list = []
    steer_list = []

    for sensor in sensors:
        speed_list.append(str_to_list(sensor[0]))
        steer_list.append(str_to_list(sensor[1]))

    return np.array(speed_list), np.array(steer_list)


def filter_folders(dir_list, folder_name, speed_min_limit=25):
    """
        Filters the data folders that meet the requirements.

        Args:
            dir_list: paths of directories,
            folder_name: the name of folder which we want the filtered folders to contain,
            speed_min_limit: the min speed the car in the video should have.
    """
    dir_list = [
        directory for directory in dir_list if directory.__contains__(folder_name.lower())]

    final_list = []
    for path in tqdm(dir_list, total=len(dir_list)):
        sensor_path = os.path.join(path, 'sensors')
        files = sorted(os.listdir(sensor_path))

        corresponds = True
        for sensor_file in files:
            sensor_file_path = os.path.join(sensor_path, sensor_file)

            sensor_data = read_csv(sensor_file_path)
            speed, _ = sensor_processing(sensor_data)

            if len(speed[speed < speed_min_limit]) > 0:
                corresponds = False
                break

        if corresponds == True:
            final_list.append(path)

    return final_list


if __name__ == '__main__':
    args = sys.argv
    path = args[1]

    # dir_paths = list_of_directories(path)
    # train_valid_test_split(dir_paths, path)
    # generate_chunks(dir_paths)

    # filtered_paths = filter_folders(
    #     dir_paths, folder_name='99C94DC769B5D96E')
    # if len(filtered_paths):
    #     write_in_txt(filtered_paths, os.path.join(path, 'filtered.txt'))

    # filtered_paths = read_txt(os.path.join(path, 'filtered.txt'))
    # filtered_paths = [path.rstrip() for path in filtered_paths]

    # resize_and_save_images(filtered_paths)
    # train_valid_test_split(filtered_paths, path)

    frames = next(get_frames(path, dataset=True))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('/home/anna/Desktop/vid_640.avi', fourcc, 30, (640, 640))

    for frame in frames:
        # print(frame)
        frame = np.expand_dims(frame, axis=0)
        frame = resize_frames(frame, (640, 640))
        # frame = color_correct(frame)
        frame = frame.squeeze(0).astype(np.uint8)

        writer.write(frame)

    writer.release()
