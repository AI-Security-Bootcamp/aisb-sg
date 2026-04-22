# %%

# Ensure the root directory is in the path for imports
import os
import sys

sys.path.append(os.path.abspath(".."))
from typing import Callable

# Common imports
import requests

from aisb_utils import report

print("It works!")
from day0_test import test_prerequisites

# Run the prerequisite checks
test_prerequisites()
