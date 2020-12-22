import torch
from torch.utils.data import DataLoader


def predict(test_db, model, device, batch_size=32):
    test_loader = DataLoader(
        test_db,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            s1, s2, data = batch
            ll_output, rewards = model(
                s1.to(device), s2.to(device), data.to(device))
            predictions.append(rewards)

    return predictions
