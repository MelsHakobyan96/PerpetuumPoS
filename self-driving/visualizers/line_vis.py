import os
import sys
import pygame
import numpy as np

from skimage import transform as tf
from tqdm import tqdm
sys.path.append('/home/anna/Desktop/self-driving/datasets')


from utils import read_csv
from processing import color_correct, save_images_as_video, get_frames, save_frames_as_video


# pygame.init()
# size = (1164*2, 874*2)
# # size=(1280*2, 720*2)
# pygame.display.set_caption("Data viewer")
# screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

# camera_surface = pygame.surface.Surface((1164, 874), 0, 24).convert()
# camera_surface = pygame.surface.Surface((1280, 720), 0, 24).convert()


rsrc = [[43.45456230828867, 118.00743250075844],
        [104.5055617352614, 69.46865203761757],
        [114.86050156739812, 60.83953551083698],
        [129.74572757609468, 50.48459567870026],
        [132.98164627363735, 46.38576532847949],
        [301.0336906326895, 98.16046448916306],
        [238.25686790036065, 62.56535881619311],
        [227.2547443287154, 56.30924933427718],
        [209.13359962247614, 46.817221154818526],
        [203.9561297064078, 43.5813024572758]]

rdst = [[10.822125594094452, 1.42189132706374],
        [21.177065426231174, 1.5297552836484982],
        [25.275895776451954, 1.42189132706374],
        [36.062291434927694, 1.6376192402332563],
        [40.376849698318004, 1.42189132706374],
        [11.900765159942026, -2.1376192402332563],
        [22.25570499207874, -2.1376192402332563],
        [26.785991168638553, -2.029755283648498],
        [37.033067044190524, -2.029755283648498],
        [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return int(p2) + 400, int(p1) + 437
    # return int(p2) - 100, int(p1)


def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]:
        img[row-sz:row+sz, col-sz:col+sz] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi/180.
    slip_fator = 0.0014
    steer_ratio = 15.3
    wheel_base = 2.67

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / \
        (steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * \
        np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 255, 0)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


def visualize_on_video(data_path, is_comma=True):
    data = read_csv(data_path).values.tolist()

    prev_path = None
    new_frames = []
    for i, video_path, idx, pred, target, speed in tqdm(data, total=len(data)):
        if prev_path != video_path:
            if is_comma:
                path_list = video_path.split('/')
                current_path = '/'.join(path_list[4:7])
                chunk_name = '_'.join(path_list[-1].split('_')[:2]) + '.avi'
                vid_path = os.path.join(
                    './', current_path, 'videos', chunk_name)
                
                frames = next(get_frames(vid_path, dataset=True))
                prev_path = vid_path
            else:
                path_list = video_path.split('/')
                current_path = '/'.join(path_list[5:])
                vid_path = os.path.join('./', current_path)

                frames = next(get_frames(vid_path, dataset=True))
                prev_path = video_path

        current_frame = frames[idx] if is_comma else frames[i]
        current_frame = color_correct(current_frame)
        
        if is_comma:
            draw_path_on(current_frame, speed, -target/10.0)

        draw_path_on(current_frame, speed, -pred/10.0, (255, 0, 0))

        new_frames.append(current_frame)

        # draw on
        # pygame.surfarray.blit_array(
        #     camera_surface, current_frame.swapaxes(0, 1))
        # camera_surface_2x = pygame.transform.scale2x(camera_surface)
        # screen.blit(camera_surface_2x, (0, 0))
        # pygame.display.flip()

    print('====== Saving the video ======')
    save_images_as_video(new_frames, './data/pred.avi', frame_rate=30, size=(1164, 874))
    # save_frames_as_video(new_frames, './data/pred.avi', frame_rate=30)


if __name__ == '__main__':
    visualize_on_video(
        data_path='./data/results.csv', is_comma=True)
