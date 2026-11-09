#!/usr/bin/env python3
"""Lipschitz constant of the synthetic designs, reported in Section 3.1.

For an equicorrelated block of size g, the population covariance has largest
eigenvalue 1+(g-1)rho. With p > n, the sample value L = ||X||_2^2 / n also
depends on random matrix fluctuations. This script computes the sample L on
the five synthetic seeds of configs/study_v2.json for each rho, and the
resulting change in the proximal gradient step size 1/L. No data download is
needed.

Usage:
    .venv/bin/python scripts/compute_lipschitz.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.data_v2 import make_synthetic  # noqa: E402
from lasso_study.solvers import lipschitz_constant  # noqa: E402


def main():
    config = json.loads((ROOT / "configs/study_v2.json").read_text())
    seeds = config["synthetic"]["seeds"]
    block = 20
    rows = {}
    for rho in (0.0, 0.3, 0.6, 0.9):
        values = [lipschitz_constant(make_synthetic(seed=s, rho=rho)["X"]) for s in seeds]
        rows[rho] = values
        print(f"rho={rho:.1f}  population max eigenvalue={1+(block-1)*rho:5.1f}  "
              f"sample L median={np.median(values):6.2f}  range={min(values):.2f}-{max(values):.2f}")
    ratio = np.median(rows[0.9]) / np.median(rows[0.0])
    print(f"step size 1/L shrinks by a factor of {ratio:.2f} from rho=0 to rho=0.9")


if __name__ == "__main__":
    main()
