import pytest
import os
import shutil
import tempfile
import subprocess
import sys
import stat
from pathlib import Path

# Add root directory to sys.path so we can import sfa_* modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

def on_rm_error(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    Usage : ``shutil.rmtree(path, onerror=on_rm_error)``
    """
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, onerror=on_rm_error)

@pytest.fixture
def temp_git_repo(temp_dir):
    """Create a temporary git repository."""
    cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # Initialize git repo
    subprocess.run(["git", "init"], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
    
    yield temp_dir
    
    os.chdir(cwd)

@pytest.fixture
def sfa_root():
    """Return the root directory of the toolkit."""
    return ROOT_DIR
