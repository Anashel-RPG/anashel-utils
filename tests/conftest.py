"""Shared pytest fixtures for testing infrastructure."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file in temp directory."""
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("test content")
    return test_file


@pytest.fixture
def mock_config():
    """Mock configuration object with common settings."""
    config = Mock()
    config.debug = False
    config.verbose = False
    config.output_dir = Path("output")
    config.input_dir = Path("input")
    return config


@pytest.fixture
def mock_lora_model():
    """Mock LoRA model for testing."""
    model = Mock()
    model.name = "test_model"
    model.layers = 380
    model.file_size = "142.55 MB"
    model.path = Path("test_model.safetensors")
    return model


@pytest.fixture
def sample_safetensor_data():
    """Sample data structure mimicking safetensor format."""
    return {
        "metadata": {"format": "lora", "version": "1.0"},
        "data": {"layer_1": [1, 2, 3], "layer_2": [4, 5, 6]}
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    # Add any singleton cleanup logic here if needed


@pytest.fixture
def capture_output(capsys):
    """Enhanced output capture with helper methods."""
    class OutputCapture:
        def __init__(self, capsys):
            self.capsys = capsys
        
        def get_stdout(self):
            return self.capsys.readouterr().out
        
        def get_stderr(self):
            return self.capsys.readouterr().err
        
        def get_both(self):
            captured = self.capsys.readouterr()
            return captured.out, captured.err
    
    return OutputCapture(capsys)


@pytest.fixture
def mock_torch_tensor():
    """Mock torch tensor for testing without requiring PyTorch."""
    tensor = Mock()
    tensor.shape = (10, 10)
    tensor.dtype = "float32"
    tensor.device = "cpu"
    return tensor


@pytest.fixture
def sample_merge_config():
    """Sample merge configuration for testing."""
    return {
        "main_lora": "model_a.safetensors",
        "merge_lora": "model_b.safetensors",
        "strategy": "adaptive",
        "weight": 0.5,
        "output_name": "merged_model"
    }