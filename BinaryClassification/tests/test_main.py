import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from BinaryClassification.main import parse_args, main

@pytest.fixture
def config_file(tmp_path):
    config = {
        "batchsize": [64],
        "lr": [0.01],
        "n_epochs": 1,
        "l2_sum_lambda": [0.0],
        "l2_mul_lambda": [0.0],
        "wn": [None],
        "depth_normalization": [False],
        "features_normalization": [None],
        "batch_norm": [False],
        "bias": [True],
        "opt_name": ["adam"],
        "device": "cpu",
        "seed": [42],
        "mislabel_percentage": 0.0
    }
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return str(config_path)

def test_parse_args_with_config_file(config_file):
    test_args = ["main.py", "--config", config_file]
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.batchsize == [64]
        assert args.lr == [0.01]
        assert args.n_epochs == 1

@patch("BinaryClassification.main.experiment")
@patch("BinaryClassification.main.analysis.analyze_results")
def test_main(mock_analyze_results, mock_experiment, config_file):
    test_args = ["main.py", "--config", config_file]
    with patch.object(sys, 'argv', test_args):
        main()
        mock_experiment.assert_called()
        mock_analyze_results.assert_called()

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])