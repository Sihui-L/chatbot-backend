from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.
    We use this to load our test environment variables.
    This runs in the same Python session as the tests,
    before any test modules or fixtures are imported.
    """
    print("\n------------ Loading test environment ------------")
    test_env_path = Path(__file__).parent / "test.env"
    if not load_dotenv(str(test_env_path), override=True):
        raise RuntimeError(
            f"Failed to load test environment variables from {test_env_path}"
        )
    print(f"Test environment loaded from {test_env_path}")
    print("--------------------------------------------------\n")


# You can add more configuration or fixtures here if needed
