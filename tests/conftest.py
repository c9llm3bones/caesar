"""Pytest configuration and fixtures for CAESAR tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import pickle
import sys
import os

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
SALAD_DIR = REPO_ROOT / "salad"

# Add SALAD to path if it exists locally
if SALAD_DIR.exists():
    sys.path.insert(0, str(SALAD_DIR))

# Add caesar to path
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="session")
def torch_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def protein_data():
    """Load reference protein structure for all tests."""
    path = TEST_DIR / "data" / "test_structure.npz"
    data = dict(np.load(path))
    return data


@pytest.fixture(scope="session")
def atol():
    return 1e-5


@pytest.fixture(scope="session")
def rtol():
    return 1e-5