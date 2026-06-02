"""Make the repository root importable so `import krypton` works under both
`pytest` and `python -m pytest`, regardless of the working directory."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
