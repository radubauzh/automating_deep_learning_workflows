import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from BinaryClassification.experiment_utils_mc import (
    product_dict, extract_best_results, mislabel_dataset, compute_weight_ranks,
    train_epoch, evaluate_model, compute_margins, set_seed, experiment
)
from BinaryClassification.model import UnderparameterizedCNN

def test_product_dict():
    result = list(product_dict(a=[1, 2], b=[3, 4]))
    assert result == [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]

def test_extract_best_results():
    import pandas as pd
    data = {
        'experiment_type': ['A', 'A', 'B', 'B'],
        'test_accuracies': [[60, 65, 70], [80, 85, 88], [50, 55, 60], [70, 75, 80]]
    }
    df = pd.DataFrame(data)
    df['final_test_accuracy'] = df['test_accuracies'].apply(lambda x: max(x) if len(x) > 0 else 0)
    best_results = extract_best_results(df)
    assert best_results['final_test_accuracy'].tolist() == [88, 80]

def test_mislabel_dataset():
    class DummyDataset:
        def __init__(self):
            self.targets = [1, 1, -1, -1]

    dataset = DummyDataset()
    mislabel_dataset(dataset, 0.5, seed=42)
    assert dataset.targets.count(1) == 2
    assert dataset.targets.count(-1) == 2

def test_compute_weight_ranks():
    model = UnderparameterizedCNN()
    ranks = compute_weight_ranks(model)
    assert isinstance(ranks, dict)

def test_set_seed():
    set_seed(42)
    assert np.random.randint(0, 100) == 51
    assert torch.randint(0, 100, (1,)).item() == 42

@patch("torch.optim.Adam")
def test_train_epoch(mock_adam):
    model = UnderparameterizedCNN()
    optimizer = mock_adam(model.parameters())
    data = torch.randn(10, 3, 32, 32)
    target = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=2)
    cfg = {
        'device': 'cpu',
        'l2_sum_lambda': 0.0,
        'l2_mul_lambda': 0.0,
        'epoch': 1,
        'depth_normalization': False,
        'features_normalization': None
    }
    results = train_epoch(model, cfg, loader, optimizer)
    assert 'loss_out' in results

@patch("torch.optim.Adam")
def test_evaluate_model(mock_adam):
    model = UnderparameterizedCNN()
    optimizer = mock_adam(model.parameters())
    data = torch.randn(10, 3, 32, 32)
    target = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=2)
    cfg = {'device': 'cpu'}
    results = evaluate_model(model, cfg, loader)
    assert 'accuracy' in results

def test_compute_margins():
    model = UnderparameterizedCNN()
    data = torch.randn(10, 3, 32, 32)
    target = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=2)
    cfg = {'device': 'cpu'}
    margins = compute_margins(model, cfg, loader)
    assert len(margins) == 10

@patch("torch.optim.Adam.step", autospec=True)
def test_experiment(mock_adam_step):
    model = UnderparameterizedCNN()
    optimizer = torch.optim.Adam(model.parameters())
    data = torch.randn(10, 3, 32, 32)
    target = torch.randint(0, 2, (10,))
    dataset = TensorDataset(data, target)
    train_loader = DataLoader(dataset, batch_size=2)
    test_loader = DataLoader(dataset, batch_size=2)
    cfg = {
        'device': 'cpu',
        'batchsize': 2,
        'lr': 0.01,
        'n_epochs': 1,
        'l2_sum_lambda': 0.0,
        'l2_mul_lambda': 0.0,
        'wn': None,
        'depth_normalization': False,
        'features_normalization': None,
        'batch_norm': False,
        'bias': True,
        'opt_name': 'adam',
        'seed': 42,
        'mislabel_percentage': 0.0,
        'in_channels': 3  # Ensure 'in_channels' is included in the configuration
    }
    results = experiment(cfg, train_loader, test_loader)
    assert 'train_epoch_losses' in results

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])