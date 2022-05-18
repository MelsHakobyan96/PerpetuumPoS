from os.path import isfile
import pandas as pd
import matplotlib.pyplot as plt
import time
import vlc
import os

from vis_utils import *
os.add_dll_directory(os.getcwd())


def play_video(source):
    # creating a vlc instance
    vlc_instance = vlc.Instance()

    # creating a media player
    player = vlc_instance.media_player_new()

    # creating a media
    media = vlc_instance.media_new(source)

    # setting media to the player
    player.set_media(media)

    # play the video
    player.play()

    # wait time
    time.sleep(0.5)

    # getting the duration of the video
    duration = player.get_length()


def visualize(one_instance_path):

    if not isfile(one_instance_path + 'vid.mp4') or not isfile(one_instance_path + 'preds.csv'):
        video_concat(one_instance_path)
        # csv_concat(one_instance_path_data)

    preds = pd.read_csv(one_instance_path + 'preds.csv')
    # pred = pd.read_csv(one_instance_path_pred + 'pred.csv')
    # for i, row in preds.iterrows():
    #     # speed = np.mean(ast.literal_eval(row['speed']))
    #     steering = np.mean(ast.literal_eval(row['steering']))
    #     # sensors.at[i, 'speed'] = speed
    #     preds.at[i, 'steering'] = steering

    plt.ion()
    figure, ax = plt.subplots()
    ax.plot(preds.index.tolist(), preds['prediction'].tolist())
    ax.plot(preds.index.tolist(), preds['target'].tolist())
    line1 = ax.axvline(x=0, color='k', linestyle='--')
    plt.title("Steering", fontsize=20)
    play_video(one_instance_path + 'vid.mp4')
    for i in range(1200):

        line1.set_xdata(i)
        figure.canvas.draw()

        figure.canvas.flush_events()

        # time.sleep(0.00000000005)


visualize("d:\\comma2k19_sensors\\data\\99c94dc769b5d96e_2018-05-01--10-47-27\\27\\")
