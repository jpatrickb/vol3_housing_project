"""
Compatibility shim so existing notebooks that import `data_loader` keep working
after moving implementation to src/data_loader.py
"""
from src.data_loader import *  # noqa: F401,F403

