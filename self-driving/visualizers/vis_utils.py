import os
import re
import cv2
import pandas as pd
import numpy as np

from moviepy.editor import VideoFileClip, concatenate_videoclips


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def video_concat(one_instance_path):
    print(one_instance_path)
    onlyfiles = [f for f in os.listdir(
        os.path.join(one_instance_path, 'videos'))]
    onlyfiles.sort(key=natural_keys)
    video_objects = [VideoFileClip(os.path.join(
        one_instance_path, 'videos', f)) for f in onlyfiles]
    final_clip = concatenate_videoclips(video_objects)
    final_clip.write_videofile(os.path.join(
        one_instance_path, 'vid.mp4'), codec='libx264')


def csv_concat(one_instance_path):
    onlyfiles = [f for f in os.listdir(one_instance_path + 'sensors')]
    onlyfiles.sort(key=natural_keys)
    combined_csv_data = pd.concat(
        [pd.read_csv(one_instance_path + 'sensors/' + f) for f in onlyfiles])
    combined_csv_data.to_csv(one_instance_path + 'csv.csv')


def cv2_resize_by_height(img, height):
    ratio = height / img.shape[0]
    width = ratio * img.shape[1]
    height, width = int(round(height)), int(round(width))
    return cv2.resize(img, (width, height))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)/2)[:2]
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    return result


def overlay_image(l_img, s_img, x_offset, y_offset):
    assert y_offset + s_img.shape[0] <= l_img.shape[0]
    assert x_offset + s_img.shape[1] <= l_img.shape[1]

    l_img = l_img.copy()
    for c in range(0, 3):
        l_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1], c] = (
                  s_img[:, :, c] * (s_img[:, :, 3]/255.0) +
                  l_img[y_offset:y_offset+s_img.shape[0],
                        x_offset:x_offset+s_img.shape[1], c] *
                  (1.0 - s_img[:, :, 3]/255.0))
    return l_img


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def reverse_signs(data):
    return [-i for i in data]
