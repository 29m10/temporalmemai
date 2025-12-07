# temporalmemai/__init__.py

from dotenv import load_dotenv

# Load .env when the package is imported
load_dotenv()

from .memory import Memory  # noqa: E402

__all__ = ["Memory"]
__version__ = "0.1.0"
