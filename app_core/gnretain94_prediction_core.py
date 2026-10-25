"""Backward-compatible import alias for the GNRetain cleaned-221 runtime."""
from gnretain221_prediction_core import *  # noqa: F401,F403

if __name__ == "__main__":
    from gnretain221_prediction_core import main
    raise SystemExit(main())
