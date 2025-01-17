import os

_TEST_ROOT = os.path.dirname(__file__)  # Root of the test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # Root of the project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # Root of the data folder

# Validate that expected paths exist
if not os.path.exists(_PROJECT_ROOT):
    raise RuntimeError(f"Project root not found: {_PROJECT_ROOT}")
if not os.path.exists(_PATH_DATA):
    raise RuntimeError(f"Data folder not found: {_PATH_DATA}")


# Helper function to get paths
def get_test_paths():
    """Return important paths for the test suite."""
    return {
        "test_root": _TEST_ROOT,
        "project_root": _PROJECT_ROOT,
        "data_path": _PATH_DATA,
    }
