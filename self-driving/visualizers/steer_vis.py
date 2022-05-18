import io
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

from tqdm import tqdm

from vis_utils import cv2_resize_by_height, rotate_image, overlay_image, smooth, reverse_signs

sys.path.append('/home/anna/Desktop/self-driving/datasets')
from utils import read_csv


def visualize(data_path, video_path, out_path, perform_smoothing=False, reverse_steer_signs=False, frame_count_limit=None):
    data = read_csv(data_path)
    human_steering = data['target']
    machine_steering = data['prediction']

    if reverse_steer_signs:
        human_steering = reverse_signs(human_steering)
        machine_steering = reverse_signs(machine_steering)

    if perform_smoothing:
        machine_steering = list(smooth(np.array(machine_steering), 20))

    steering_min = min(np.min(human_steering), np.min(machine_steering))
    steering_max = max(np.max(human_steering), np.max(machine_steering))

    cap = cv2.VideoCapture(video_path)

    vid_size = (1164, 874)

    vw = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'XVID'), 30, vid_size)
    w, h = vid_size

    for f_cur in tqdm(range(len(machine_steering)), total=len(machine_steering)):
        if (frame_count_limit is not None) and (f_cur >= frame_count_limit):
            break

        rret, rimg = cap.read()
        assert rret

        dimg = rimg.copy()
        dimg[:] = (0, 0, 0)

        ry0, rh = 80, 500
        dimg = dimg[100:, :930]
        dimg = cv2_resize_by_height(dimg, h-rh)

        fimg = rimg.copy()
        fimg[:] = (0, 0, 0)
        fimg[:rh] = rimg[ry0:ry0+rh]
        dh, dw = dimg.shape[:2]
        fimg[rh:, :dw] = dimg[:]

        ########################## plot ##########################
        plot_size = (500, dh)
        win_before, win_after = 150, 150

        xx, hh, mm = [], [], []
        for f_rel in range(-win_before, win_after+1):
            f_abs = f_cur + f_rel
            if f_abs < 0 or f_abs >= len(machine_steering):
                continue
            xx.append(f_rel/30)
            hh.append(human_steering[f_abs])
            mm.append(machine_steering[f_abs])

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        steering_range = max(abs(steering_min), abs(steering_max))
        ylim = [-steering_range, steering_range]

        axis.set_xlabel('Current Time (secs)')
        axis.set_ylabel('Steering Angle')
        axis.axvline(x=0, color='k', ls='dashed')
        axis.plot(xx, hh)
        axis.plot(xx, mm)
        axis.set_xlim([-win_before/30, win_after/30])
        axis.set_ylim(ylim)
        axis.label_outer()

        buf = io.BytesIO()
        # http://stackoverflow.com/a/4306340/627517
        sx, sy = plot_size
        sx, sy = round(sx / 100, 1), round(sy / 100, 1)

        fig.set_size_inches(sx, sy)
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        buf_img = PIL.Image.open(buf)
        pimg = np.asarray(buf_img)
        plt.close(fig)

        pimg = cv2.resize(pimg, plot_size)
        pimg = pimg[:, :, :3]
        ph, pw = pimg.shape[:2]
        pimg = 255 - pimg
        fimg[rh:, -pw:] = pimg[:]

        ####################### human steering wheels ######################
        wimg = cv2.imread("./images/wheel-tesla-image-150.png",
                          cv2.IMREAD_UNCHANGED)

        human_wimg = rotate_image(wimg, -human_steering[f_cur])
        fimg = overlay_image(
            fimg, human_wimg, y_offset=rh+50, x_offset=dw-300)

        ####################### machine steering wheels ######################
        disagreement = abs(machine_steering[f_cur] - human_steering[f_cur])
        machine_wimg = rotate_image(wimg, -machine_steering[f_cur])

        red_machine_wimg = machine_wimg.copy()
        green_machine_wimg = machine_wimg.copy()
        red_machine_wimg[:, :, 2] = 255
        green_machine_wimg[:, :, 1] = 255

        max_disagreement = 10
        r = min(1., disagreement / max_disagreement)
        g = 1 - r
        assert r >= 0
        assert g <= 1

        machine_wimg = cv2.addWeighted(
            red_machine_wimg, r, green_machine_wimg, g, 0)
        fimg = overlay_image(
            fimg, machine_wimg, y_offset=rh+50, x_offset=dw-100)

        vw.write(fimg)

    cap.release()
    vw.release()


if __name__ == '__main__':
    data_path = './data/results.csv'
    video_path = './data/99c94dc769b5d96e_2018-05-01--10-47-27/27/vid.mp4'
    out_path = './data/vid_pred.avi'
    visualize(data_path, video_path, out_path,
              perform_smoothing=True,
              reverse_steer_signs=True)
