import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from BinaryClassification.GUI import ConfigGenerator

@pytest.fixture
def app():
    """Create QApplication instance for PyQt tests."""
    _app = QApplication.instance()
    if not _app:
        _app = QApplication([])
    return _app

@pytest.fixture
def gui(app):
    """Create the GUI instance with test inputs."""
    window = ConfigGenerator()
    return window

def test_validate_positive_int(gui):
    assert gui.validate_positive_int("5") == 5
    assert gui.validate_positive_int("-1") is None
    assert gui.validate_positive_int("abc") is None

def test_validate_non_negative_float(gui):
    assert gui.validate_non_negative_float("0.5") == 0.5
    assert gui.validate_non_negative_float("-0.5") is None
    assert gui.validate_non_negative_float("xyz") is None

def test_validate_comma_separated_list_integers(gui):
    result = gui.validate_comma_separated_list("1,2,3", gui.validate_positive_int)
    assert result == [1, 2, 3]
    assert gui.validate_comma_separated_list("1,a", gui.validate_positive_int) is None

def test_validate_comma_separated_list_floats(gui):
    result = gui.validate_comma_separated_list("0.1,0.3", gui.validate_non_negative_float)
    assert result == [0.1, 0.3]
    assert gui.validate_comma_separated_list("0.1,-0.2", gui.validate_non_negative_float) is None

@patch("BinaryClassification.GUI.QProcess", autospec=True)
def test_create_json_and_run_script(mock_qprocess, gui, tmp_path):
    """Test JSON creation and script invocation without running a real subprocess."""
    mock_instance = mock_qprocess.return_value
    mock_instance.start = MagicMock()
    mock_instance.finished = MagicMock()
    mock_instance.readyRead = MagicMock()
    mock_instance.errorOccurred = MagicMock()
    
    gui.batchsize_entry.setText("64")
    gui.lr_entry.setText("0.01")
    gui.n_epochs_entry.setText("2")
    gui.l2_sum_lambda_entry.setText("")
    gui.l2_mul_lambda_entry.setText("")
    gui.wn_entry.setText("0.8")
    gui.seeds_entry.setText("42")
    gui.create_json()
    
    # Check that JSON was created
    config_dir = os.path.join(os.path.dirname(__file__), "..", "Configs")
    created_files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
    assert len(created_files) > 0
    
    # Verify that the process started
    mock_instance.start.assert_called_once()
    
    # Load and inspect generated config for correctness
    with open(os.path.join(config_dir, created_files[0]), "r") as f:
        data = json.load(f)
        assert data["batchsize"] == 64
        assert data["lr"] == [0.01]
        assert data["n_epochs"] == 2

if __name__ == "__main__":
    pytest.main()