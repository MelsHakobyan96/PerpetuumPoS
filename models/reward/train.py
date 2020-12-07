import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.reward.module import RewardPredictor


def train(train_db, batch_size, device=None, lr=3e-4, save=False, path='./logs/model.pth'):
    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    cnn_inp_size = (batch_size * 3) // 2
    model = RewardPredictor(cnn_inp_size, train_db.mlp_inp_size)
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    total_loss = 0
    for batch in train_loader:
        s1, s2, data, target = batch
        optimizer.zero_grad()
        target = target[0].unsqueeze(0).T

        predictions, reward = model(s1.to(device), s2.to(device), data.to(device))
        loss = criterion(predictions, target.to(device))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if save:
        torch.save(model.state_dict(), path)

    return total_loss
