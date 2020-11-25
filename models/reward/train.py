from module import RewardPredictor
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


def train(train_db, batch_size, device=None, lr=3e-4):
    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    model = RewardPredictor(*train_db.sizes)
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    total_loss = 0
    for batch in train_loader:
        s1, s2, data, target = batch

        optimizer.zero_grad()

        predictions = model(s1.to(device), s2.to(device), data.to(device))
        # TODO
        # Make target and prediction to match
        loss = criterion(predictions, target.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss
