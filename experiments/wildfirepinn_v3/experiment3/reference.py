"""Hidden reference fires. Training code does not import this module."""

from __future__ import annotations

import time

import numpy as np

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.solver import propagate


def circle_problem(n: int, x0: float = 200.0, y0: float = -150.0) -> CircleProblem:
    return CircleProblem(n=n, x0=x0, y0=y0)


def make_reference(problem: CircleProblem, times: list[float]) -> dict:
    x, y = problem.grid()
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
    rate = spatial_rate_np(xx, yy)
    started = time.perf_counter()
    fields = propagate(phi0, x, y, rate, times)
    elapsed = time.perf_counter() - started
    return {
        "problem": problem,
        "x": x,
        "y": y,
        "phi0": phi0,
        "rate_true_eval_only": rate,
        "fields": fields,
        "solver_seconds": elapsed,
        "times": [float(t) for t in times],
    }
