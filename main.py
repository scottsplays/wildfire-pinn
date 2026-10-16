"""Entry point for the wildfire level-set PINN.

    python main.py synthetic --epochs 2001
    python main.py demo
    python main.py pipeline --epochs 4000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wildfire_pinn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
