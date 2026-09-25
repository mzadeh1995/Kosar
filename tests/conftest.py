# ==============================================================================
# tests/conftest.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Session-wide guard. pytest imports conftest.py before any test module,
# regardless of collection order, so this runs before xgboost/pomegranate load.
# Direct assignment on purpose (not setdefault): the test session must be
# single-threaded for OpenMP, removing both the libomp conflict and any
# dependence on shell-exported values.
# ==============================================================================

import os

# Prevent XGBoost/libomp and pomegranate HMM from crashing the shared pytest process.
os.environ["OMP_NUM_THREADS"] = "1"
