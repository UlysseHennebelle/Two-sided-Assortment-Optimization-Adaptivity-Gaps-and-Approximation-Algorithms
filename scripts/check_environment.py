"""Verify the project interpreter, Gurobi license, and required packages."""

from __future__ import annotations

import importlib.util
import sys

import gurobipy as gp


def main() -> None:
    required = ["numpy", "scipy", "pandas", "pyarrow", "cvxpy", "matplotlib", "pytest"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing packages: {', '.join(missing)}")
    model = gp.Model("tsao_environment_check")
    model.Params.OutputFlag = 0
    x = model.addVar(lb=0.0, ub=1.0)
    model.setObjective(x, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL or abs(model.ObjVal - 1.0) > 1e-12:
        raise SystemExit("Gurobi smoke solve failed")
    print(f"python={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print("gurobi=" + ".".join(str(item) for item in gp.gurobi.version()))
    print("environment_ok=true")


if __name__ == "__main__":
    main()
