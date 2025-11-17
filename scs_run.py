#!/usr/bin/env python3
"""
Standalone script to run SCS calibration.

Can be invoked from CI/benchmark pipelines.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scs.cli import main

if __name__ == "__main__":
    sys.exit(main())
