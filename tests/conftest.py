"""Pytest configuration and fixtures for CAESAR tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import random
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

from caesar.modules.utils.collections import deepcopy

from tests.configs import test_deterministic
@pytest.fixture
def cfg():
    return deepcopy(test_deterministic)
                    
@pytest.fixture(scope="function")
def seed():
    # this seed stands as an initializer for torch & jax rngs
    return 0

@pytest.fixture(autouse=True, scope="function")
def fixed_seed(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if cuda
    yield

@pytest.fixture
def jax_keys(seed):
    try:
        import jax
    except Exception as e:
        pytest.skip(f"JAX not available: {e}")

    key = jax.random.PRNGKey(seed)
    key_init, key_apply = jax.random.split(key, 2)
    return key_init, key_apply

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
