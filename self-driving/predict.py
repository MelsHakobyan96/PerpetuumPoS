import os
from cv2 import resize
import torch
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from datasets.comma import Comma2k19Dataset
from datasets.processing import get_frames, resize_frames
from mlp_mixer_pytorch import MLPMixer


def save_meta(data, path, is_comma):
    dir_path = '__'.join(data[0][0].split('/')[5:-2]) if is_comma else path

    meta_file = os.path.join(path, f'{dir_path}.csv')
    df = pd.DataFrame(
        data, columns=['path', 'frame_id', 'prediction', 'target', 'speed'])
    df.to_csv(meta_file)


def predict(checkpoint_path, test_path, data_dir_path=None, batch_size=1, is_comma=True, cuda=True):
    device = torch.device("cuda" if cuda is True else "cpu")
    print(f'Testing the model on {device} device.\n')

    results_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    model = MLPMixer(
        image_size=(160, 320),
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=1
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        if is_comma:
            predict_from_loader(data_dir_path, test_path,
                                model, device, batch_size,
                                results_path)
        else:
            predict_frame_by_frame(test_path, model,
                                   batch_size, device,
                                   results_path)


def predict_frame_by_frame(video_path, model, batch_size, device, results_path):
    video_frames = torch.tensor(next(get_frames(video_path, dataset=True)))
    test_data = TensorDataset(video_frames)
    test_loader = DataLoader(test_data, batch_size=8 *
                             batch_size, drop_last=True)

    test_tqdm = tqdm(test_loader, total=len(test_loader),
                     desc=f"Testing progress:")

    video_outputs = []
    for frames in test_tqdm:
        frames = resize_frames(frames[0], size=(160, 320))

        frames = torch.FloatTensor(frames)
        frames = torch.reshape(
            frames, (8*batch_size, 3, 160, 320)).to(device)

        preds = model(frames)
        outputs = [(video_path, i, preds[i].item(), None, 70)
                   for i in list(range(frames.shape[0]))]

        video_outputs.extend(outputs)

    save_meta(video_outputs, results_path, is_comma=False)


def predict_from_loader(data_dir_path, test_csv_path, model, device, batch_size, results_path):
    test_data = Comma2k19Dataset(data_dir_path, test_csv_path, is_testing=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    test_tqdm = tqdm(enumerate(test_loader), total=len(test_loader),
                     desc=f"Testing progress:")

    video_outputs = []
    for idx, (frames, targets, speed, is_new_video, path) in test_tqdm:
        if any(is_new_video) and idx != 0:
            save_meta(video_outputs, results_path, is_comma=True)
            video_outputs = []

        frames = torch.reshape(
            frames, (8*batch_size, 3, 160, 320)).to(device)
        targets = torch.reshape(
            targets, (-1, 1)).float().squeeze(0).to(device)
        speed = torch.reshape(
            speed, (-1, 1)).float().squeeze(0)

        preds = model(frames)
        outputs = [(path[0], i, preds[i].item(), targets[i].item(), speed[i].item())
                   for i in list(range(frames.shape[0]))]
        video_outputs.extend(outputs)


if __name__ == '__main__':
    checkpoint = '/home/selfdriving/Desktop/self-driving/checkpoints/epoch_22.pth'
    data_dir_path = '/media/external_disk/comma2k19_sensors/'
    # test_path = 'paths/test.csv'
    test_path = 'data/third.mp4'
    is_comma = False

    predict(checkpoint_path=checkpoint,
            data_dir_path=data_dir_path,
            test_path=test_path,
            is_comma=is_comma)
