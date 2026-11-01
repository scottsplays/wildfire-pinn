"""Start the WildfirePINN inference service (model_type "pinn") on port 8082.

    cd ml_pinn_service
    python serve.py --port 8082

The code lives in the pinn_service/ package next to this file; see README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinn_service.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
