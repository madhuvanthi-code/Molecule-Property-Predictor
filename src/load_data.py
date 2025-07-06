from torch_geometric.datasets import QM9
from torch_geometric.transforms import Complete
from torch.utils.data import random_split
from src.normalize import normalize_targets

def get_datasets(path):
    dataset = QM9(root=path, transform=Complete())
    dataset, scaler = normalize_targets(dataset, target_index=0)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size]) + (scaler,)