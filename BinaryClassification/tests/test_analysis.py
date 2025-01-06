import os
import pytest
import pandas as pd
import tempfile
import pickle
import shutil
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from BinaryClassification import analysis

def test_flatten_list():
    nested = [[1, 2], [3, 4]]
    result = analysis.flatten_list(nested)
    assert result == [1, 2, 3, 4]

def test_flatten_list_empty():
    nested = []
    result = analysis.flatten_list(nested)
    assert result == []

def test_get_last_n_values():
    values = [1, 2, 3, 4, 5]
    assert analysis.get_last_n_values(values, 2) == [4, 5]

def test_get_last_n_values_nested():
    nested_values = [[1, 2, 3], [4, 5, 6]]
    truncated = analysis.get_last_n_values(nested_values, 2)
    assert truncated == [[2, 3], [5, 6]]

def test_get_last_n_values_series():
    series = pd.Series([[1, 2, 3], [4, 5, 6]])
    truncated = analysis.get_last_n_values(series, 2)
    assert truncated.tolist() == [[2, 3], [5, 6]]

@patch("BinaryClassification.analysis.generate_gpt_analysis_report", autospec=True)
def test_analyze_results(mock_gpt):
    # Create dummy results
    dummy_results = {
        "test_accuracies": [[60, 65, 70], [80, 85, 88]],
        "test_losses": [[0.9, 0.8, 0.6], [0.5, 0.4, 0.3]],
        "train_epoch_losses": [[1.2, 1.0], [1.1, 1.0]],
        # Add other necessary keys here
    }
    temp_dir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(temp_dir, "test_results.pkl")
        with open(test_file, "wb") as f:
            pickle.dump(dummy_results, f)

        # Ensure no exception is thrown
        analysis.analyze_results(test_file)
        mock_gpt.assert_called_once()
    finally:
        shutil.rmtree(temp_dir)

@patch("openai.ChatCompletion.create", autospec=True)
def test_generate_gpt_analysis_report(mock_openai):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked GPT analysis report"
    mock_openai.return_value = mock_response
    
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "test_summary.csv")
        df = pd.DataFrame({
            "test_accuracies": [[60, 65, 70], [80, 85, 88]],
            "test_losses": [[0.9, 0.8, 0.6], [0.5, 0.4, 0.3]],
        })
        df.to_csv(csv_path, index=False)

        analysis.generate_gpt_analysis_report(csv_path, output_folder=temp_dir)
        output_path = os.path.join(temp_dir, "analysis_report.md")
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Check for key sections in the report
            assert "Overall Performance" in content
            assert "Best Parameters" in content
            assert "Experiment Type Analysis" in content
            assert "Top Experiments" in content
            assert "Detailed Insights" in content
            assert "Recommendations" in content
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])