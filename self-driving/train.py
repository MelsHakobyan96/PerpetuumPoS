import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from termcolor import colored

from datasets.comma import Comma2k19Dataset
from models.optimizer import get_optimizer

from mlp_mixer_pytorch import MLPMixer


def train_model(args):
    epochs = args.epochs
    batch_size = args.batch_size

    print(args.cuda)
    device = torch.device("cuda" if args.cuda is True else "cpu")
    print(colored(f'Training the model on {device} device.', 'green'))

    train_data = Comma2k19Dataset(args.data_dir_path, args.train_csv_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    valid_data = Comma2k19Dataset(args.data_dir_path, args.valid_csv_path)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

    current_dir = os.getcwd()
    checkpoints_path = os.path.join(current_dir, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    history_path = os.path.join(current_dir, 'history')
    if not os.path.exists(history_path):
        os.mkdir(history_path)

    history_file = os.path.join(history_path, 'history.csv')

    model = MLPMixer(
        image_size=(160, 320),
        channels=3,
        patch_size=16,
        dim=512,
        depth=6,
        num_classes=1
    )
    opt = get_optimizer(model)
    mse_loss = nn.MSELoss()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    model = model.to(device)

    best_loss = 10
    best_checkpoint_path = os.path.join(checkpoints_path, 'best_model_depth_6.pth')

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_tqdm = tqdm(train_loader, total=len(train_loader),
                          desc=f"Training progress for epoch {epoch + 1}")

        for frames, targets in train_tqdm:
            frames = torch.reshape(
                frames, (8*frames.shape[0], 3, 160, 320)).to(device)
            targets = torch.reshape(
                targets, (-1, 1)).float().squeeze(0).to(device)

            opt.zero_grad()

            outputs = model(frames)
            loss = mse_loss(outputs.squeeze(0), targets)

            loss.backward(retain_graph=True)
            opt.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # validate
        val_loss = 0
        with torch.no_grad():
            valid_tqdm = tqdm(valid_loader, total=len(
                valid_loader), desc=f"Validation progress for epoch {epoch + 1}")

            for frames, targets in valid_tqdm:
                frames, targets = frames.squeeze(0).to(
                    device), targets.float().squeeze(0).to(device)

                outputs = model(frames)
                val_loss += mse_loss(outputs.squeeze(0), targets).item()

        epoch_val_loss = val_loss / len(valid_loader)

        print(
            colored(f'epoch: {epoch + 1}, training loss: {epoch_train_loss}', 'yellow'))
        print(
            colored(f'epoch: {epoch + 1}, validation loss: {epoch_val_loss}', 'yellow'))

        if epoch_val_loss < best_loss:
            print(colored(
                "Saving the best model that shows smallest loss on validation set.", 'green'))
            best_loss = epoch_val_loss
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.optimizer.state_dict()},
                       best_checkpoint_path)

        with open(history_file, 'a', newline='') as history_f:
            writer = csv.writer(history_f, delimiter=',')
            if epoch == 0:
                writer.writerow(['epoch', 'training_loss', 'validation_loss'])
            writer.writerow([epoch, epoch_train_loss, epoch_val_loss])
