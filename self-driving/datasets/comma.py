import os
import sys
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from .utils import *
from .processing import get_frames


class Comma2k19Dataset(Dataset):
    """Comma2k19 dataset."""

    def __init__(self, dir_path, csv_path, is_testing=False):
        """
        Args:
            data_path: the path to where the data is located,
        """
        self.dir_path = dir_path
        csv_path = os.path.join(self.dir_path, csv_path)
        self.data = read_csv(csv_path)
        self.is_testing = is_testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Getting the paths of the given idx from a csv file, and getting the frames and targets for the steering.
        """
        video_path, sensor_path = self.data.iloc[idx].values
        video_path = os.path.join(self.dir_path, video_path)
        sensor_path = os.path.join(self.dir_path, sensor_path)

        # checking whether the chunk is from new video or not
        is_new_video = True if video_path.__contains__('_0') else False

        # a generator object, might change it soon
        # processing frames of the video
        frames = next(get_frames(video_path, dataset=True))
        frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)

        # processing sensor data
        sensors = read_csv(sensor_path)
        steer = sensors['steering'].apply(str_to_list)
        steer = torch.FloatTensor(steer).mean(dim=1).unsqueeze(1)

        if self.is_testing:
            speed = sensors['speed'].apply(str_to_list)
            speed = torch.FloatTensor(speed).mean(dim=1).unsqueeze(1)

            return frames, steer, speed, is_new_video, video_path

        return frames, steer


class CommaAIDataset(Dataset):
    """Comma-ai dataset."""

    def __init__(self, dir_path, file_name):
        """
        Args:
            data_path: the path to where the data is located,
        """
        self.dir_path = dir_path
        camera_path = os.path.join(self.dir_path, f'camera/{file_name}')
        log_path = os.path.join(self.dir_path, f'log/{file_name}')

        self.frames = convert_hdf5_to_numpy(camera_path)
        self.logs = convert_hdf5_to_numpy(log_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return


if __name__ == '__main__':
    args = sys.argv
    dir_path = args[1]

    # file_name = '2016-01-31--19-19-25.h5'
    # comma_dataset = CommaAIDataset(dir_path, file_name)

    comma_dataset = Comma2k19Dataset(dir_path, 'paths/train.csv')
    x = comma_dataset[0]
