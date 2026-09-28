"""NIRYUKTI: independent sparse optimization, with the legacy vantage API retained."""
from vantage import (Model, solve, SolverSession, NativeSession, diagnose_infeasibility,
                     propose_repair, rhs_sensitivity, stable_plan_model, replan)
__all__ = ["Model", "solve", "SolverSession", "NativeSession", "diagnose_infeasibility",
           "propose_repair", "rhs_sensitivity", "stable_plan_model", "replan"]

from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("niryukti")
except PackageNotFoundError:
    __version__ = "0.2.2"  # Source checkout; keep aligned with project metadata.
