import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

@torch.no_grad()
def evaluate(model, test_data, scaler):
    model.eval()
    loader = DataLoader(test_data, batch_size=32)
    all_preds, all_labels = [], []

    for batch in loader:
        pred = model(batch).squeeze().cpu()
        label = batch.y.squeeze().cpu()
        all_preds.extend(pred.tolist())
        all_labels.extend(label.tolist())

    all_preds = scaler.inverse_transform([[p] for p in all_preds])
    all_labels = scaler.inverse_transform([[l] for l in all_labels])

    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    print(f"Test MAE: {mae:.4f}, R²: {r2:.4f}")