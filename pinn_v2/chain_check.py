"""Short-window (chained) PINN test. Uses gate_check.py unchanged.

    python -m pinn_v2.chain_check --quick     # ~2 min: checks it runs
    python -m pinn_v2.chain_check             # contract (0-1-3-6 h) and hourly chains, seeds 0-2 (~30-40 min CPU)
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np

from pinn_service import physics
from pinn_service.forecaster import signed_distance
from pinn_v2 import gate_check as g

CHAINS = {"contract": (0, 1, 3, 6), "hourly": (0, 1, 2, 3, 4, 5, 6)}


def window_case(case, phi_start, hours):
    return g.Case(case.name, case.res, case.xs, case.ys, phi_start, case.rate, case.wind,
                  horizon_s=hours * 3600.0, notes=case.notes)


def handoff(phi_end, phi_start, res):
    # fire never un-burns; re-distance to a clean signed distance for the next window
    return signed_distance((phi_end < 0) | (phi_start < 0), res)


def train_window(wc, hours, steps, batch, seed):
    saved = g.HORIZONS_H
    g.HORIZONS_H = (hours,)              # make gate_check predict at the end of this window
    try:
        preds, meta, _ = g.train_and_predict(wc, steps, batch, seed, "local", log_every=max(1, steps // 4))
    finally:
        g.HORIZONS_H = saved
    return preds[hours], meta


def chain_run(case, chain, steps, batch, seed, out_dir):
    label = "-".join(map(str, chain))
    print(f"\n== {case.name}: chained PINN, windows {label} h, {steps} steps each, seed {seed}")
    refs = g.reference(case)
    phi = {"pinn": case.phi0, "solver_chain": case.phi0, "first_guess_chain": case.phi0}
    at = {k: {} for k in phi}
    fields, windows, started = {}, [], time.perf_counter()
    for i, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
        w = b - a
        print(f"  window {a}->{b} h")
        end, meta = train_window(window_case(case, phi["pinn"], w), w, steps, batch, seed * 100 + i)
        solver_end = physics.propagate(phi["solver_chain"], case.res, case.rate, case.wind, [w * 3600.0])[w * 3600.0]
        fg = window_case(case, phi["first_guess_chain"], w)
        phi = {"pinn": handoff(end, phi["pinn"], case.res),
               "solver_chain": handoff(solver_end, phi["solver_chain"], case.res),
               "first_guess_chain": handoff(fg.phi0 - w * 3600.0 * g.local_speed(fg), fg.phi0, case.res)}
        windows.append({"window_h": [a, b], "final_residual": meta["loss_history"][-1][1], "train_seconds": meta["train_seconds"]})
        if b in (1, 3, 6):
            for k in phi:
                at[k][b] = g.compare(phi[k], refs[b], case.phi0, case.res)
            r = at["pinn"][b]
            print(f"   +{b} h: IoU {r['iou']:.3f}, new-area {r['growth_iou']:.3f}, boundary {r['boundary_cells']:.2f} cells"
                  f"   | solver chain {at['solver_chain'][b]['iou']:.3f}, first-guess chain {at['first_guess_chain'][b]['iou']:.3f}")
            fields[b] = phi["pinn"]
    passed = g.gate_passed(at["pinn"][6])
    report = {"case": case.name, "method": f"chained_{label}", "seed": seed, "steps_per_window": steps, "gate": g.GATE,
              "passed_at_6h": passed, "horizons": at["pinn"], "solver_chain": at["solver_chain"],
              "first_guess_chain": at["first_guess_chain"], "windows": windows,
              "total_train_seconds": round(time.perf_counter() - started, 1), "notes": case.notes}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.name}_chain{label.replace('-', '')}_s{seed}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    g.figure(case, fields, refs, out_dir / f"{stem}.png")
    print(f"   GATE at 6 h: {'PASS' if passed else 'FAIL'}   (figure: {out_dir / (stem + '.png')})")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chain", nargs="+", choices=list(CHAINS), default=["contract", "hourly"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=None, help="steps per window (default: 6000 split evenly)")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--case", choices=["fuel", "circle"], default="fuel")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(g.REPO / "outputs" / "pinn_v2_gate"))
    a = p.parse_args()
    if a.quick:
        a.seeds, a.steps, a.batch = [0], 100, 2048
    out = Path(a.out) / ("chain_quick" if a.quick else "chain")
    case = {"fuel": g.fuel_case, "circle": g.circle_case}[a.case]()
    rows = []
    for name in a.chain:
        chain = CHAINS[name]
        steps = a.steps or 6000 // (len(chain) - 1)     # same 6000-step total for every chain
        for seed in a.seeds:
            r = chain_run(case, chain, steps, a.batch, seed, out)
            rows.append({"chain": name, "seed": seed, "steps_per_window": steps,
                         "iou_6h": r["horizons"][6]["iou"], "growth_iou_6h": r["horizons"][6]["growth_iou"],
                         "boundary_cells_6h": r["horizons"][6]["boundary_cells"],
                         "iou_3h": r["horizons"][3]["iou"], "iou_1h": r["horizons"][1]["iou"],
                         "solver_chain_iou_6h": r["solver_chain"][6]["iou"],
                         "first_guess_chain_iou_6h": r["first_guess_chain"][6]["iou"],
                         "train_s": r["total_train_seconds"], "gate": "PASS" if r["passed_at_6h"] else "FAIL"})
    with open(out / "chain.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n=== CHAINED PINN ({a.case}, 6 h vs unbroken solver; gate: IoU >= 0.90, new-area >= 0.70, boundary <= 2 cells)")
    print(f"{'chain':9s} {'seed':>4s} {'IoU':>6s} {'new':>6s} {'bound':>6s} {'solverCh':>8s} {'guessCh':>8s} {'train_s':>7s} gate")
    for r in rows:
        print(f"{r['chain']:9s} {r['seed']:4d} {r['iou_6h']:6.3f} {r['growth_iou_6h']:6.3f} {r['boundary_cells_6h']:6.2f} "
              f"{r['solver_chain_iou_6h']:8.3f} {r['first_guess_chain_iou_6h']:8.3f} {r['train_s']:7.0f} {r['gate']}")
    for name in a.chain:
        ious = np.array([r["iou_6h"] for r in rows if r["chain"] == name])
        sd = f" +/- {ious.std(ddof=1):.3f} (n={len(ious)})" if len(ious) > 1 else ""
        print(f"  {name}: mean 6 h IoU {ious.mean():.3f}{sd}")
    print(f"(one-window PINN: 6 h IoU 0.821 +/- 0.021; table saved to {out / 'chain.csv'})")


if __name__ == "__main__":
    main()