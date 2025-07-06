from sklearn.preprocessing import StandardScaler
import torch

def normalize_targets(dataset, target_index):
    y = torch.stack([data.y[target_index] for data in dataset]).view(-1, 1)
    scaler = StandardScaler()
    y_scaled = torch.tensor(scaler.fit_transform(y), dtype=torch.float32)
    for i, data in enumerate(dataset):
        data.y = y_scaled[i]
    return dataset, scaler