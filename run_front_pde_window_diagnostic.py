"""Same front-local PDE comparison, with time limited to passes 0 through 2.

PDE collocation times are uniform from the first pass to the third pass.
Nothing is sampled after pass 2, so the pass-2 front is not held in place
for the rest of the six-day window. Observation loss, bands, soft weights,
network, and R are otherwise the same as run_front_pde_diagnostic.py.
"""

from __future__ import annotations

import run_front_pde_diagnostic as diag

OUT_NAME = "kincade_front_pde_window"


def main() -> None:
    diag.OUT = diag.ROOT / "outputs" / OUT_NAME
    diag.OUT.mkdir(parents=True, exist_ok=True)
    prepared = diag._prepare()
    tau_hi = float(prepared["tau"][2])
    if tau_hi <= 0:
        raise RuntimeError("pass 2 has no positive normalized time")
    print(
        f"PDE tau is restricted to [0, {tau_hi:.4f}], "
        f"which is {tau_hi * prepared['duration'] / 3600:.2f} hours.",
        flush=True,
    )
    results = [
        diag._train_one(
            {"name": "observation_only", "kind": "current"},
            0.0,
            prepared,
            tau_lo=0.0,
            tau_hi=tau_hi,
        )
    ]
    for strategy in diag.STRATEGIES:
        for weight in diag.WEIGHTS:
            results.append(diag._train_one(strategy, weight, prepared, tau_lo=0.0, tau_hi=tau_hi))
    diag._write_comparison(
        results,
        prepared,
        title="PDE enforced only from pass 0 through pass 2. Front masks still use the observed SDF.",
    )
    diag._print_table(results)


if __name__ == "__main__":
    main()
