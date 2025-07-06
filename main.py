import torch
from src.load_data import get_datasets
from src.models import GCNModel
from src.train import train
from src.evaluate import evaluate

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset, scaler = get_datasets("data/qm9")
    model = GCNModel(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    train(model, train_dataset, val_dataset, optimizer, loss_fn, scaler)
    torch.save(model.state_dict(), "model.pt")
    evaluate(model, test_dataset, scaler)