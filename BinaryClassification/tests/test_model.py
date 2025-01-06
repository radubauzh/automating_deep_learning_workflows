import os
import sys
import pytest
import torch

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from BinaryClassification.model import (
    UnderparameterizedCNN, OverparameterizedCNN, SuperUnderparameterizedCNN,
    PaperWNInitModel, PaperWNModel, RhoWNModel, set_model_norm_to_one
)

def test_underparameterized_cnn():
    model = UnderparameterizedCNN()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_overparameterized_cnn():
    model = OverparameterizedCNN()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_super_underparameterized_cnn():
    model = SuperUnderparameterizedCNN()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_paper_wn_init_model():
    base_model = UnderparameterizedCNN()
    model = PaperWNInitModel(base_model)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_paper_wn_model():
    base_model = UnderparameterizedCNN()
    model = PaperWNModel(base_model)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_rho_wn_model():
    base_model = UnderparameterizedCNN()
    model = RhoWNModel(base_model)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 1)

def test_set_model_norm_to_one():
    model = UnderparameterizedCNN()
    set_model_norm_to_one(model)
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            weight_norm = torch.norm(m.weight.data, p=2)
            assert torch.isclose(weight_norm, torch.tensor(1.0), atol=1e-6)

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])