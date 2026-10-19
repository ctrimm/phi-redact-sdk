"""Make the test suite exercise the repo tree, not any installed copy."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
