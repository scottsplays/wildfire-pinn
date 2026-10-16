"""Command line for the level-set wildfire PINN.

Examples
--------
python main.py synthetic --epochs 2001
python main.py demo
python main.py pipeline --epochs 4000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from wildfire_pinn.config import DATASETS, PROJECT_ROOT, ProjectConfig
from wildfire_pinn.data_pipeline import (
    build_perimeters,
    build_spread_rate,
    build_stack,
    describe_stack,
    download_firms,
    download_gridmet,
    download_landfire,
    filter_detections,
)
from wildfire_pinn.demo import run_demo
from wildfire_pinn.pinn import (
    RealConfig,
    SyntheticConfig,
    evaluate_real,
    get_device,
    load_fire_case,
    relative_l2_synthetic,
    synthetic_snapshots,
    train_real,
    train_synthetic,
    write_metrics,
)
from wildfire_pinn.plots import plot_dashboard, plot_losses, plot_scalar_grid, plot_synthetic_fronts

log = logging.getLogger("wildfire_pinn")


def setup_logging(log_path: Path | None, *, replace: bool = True) -> None:
    logger = logging.getLogger("wildfire_pinn")
    if logger.handlers and not replace:
        return
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = logging.FileHandler(log_path, encoding="utf-8")
        handle.setFormatter(formatter)
        logger.addHandler(handle)


def key_from_file(path: Path) -> str | None:
    """First non-comment line of a key file. Comments and blank lines are ignored."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip('"').strip("'")
        if line and not line.startswith("#"):
            return line
    return None


def _firms_key(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRE_PERIMETER_API_KEY")
    if env and env.strip():
        return env.strip()
    return key_from_file(PROJECT_ROOT / "firms_key.txt")


def _require(paths: list[Path], hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        listed = "\n  ".join(str(path) for path in missing)
        raise RuntimeError(f"Missing files:\n  {listed}\n{hint}")


def _read_detections(cfg: ProjectConfig):
    import pandas as pd

    frame = pd.read_csv(cfg.firms_csv)
    return filter_detections(frame)


def cmd_synthetic(args, cfg: ProjectConfig) -> None:
    setup_logging(cfg.synthetic_dir / "run.log")
    hp = SyntheticConfig(epochs=args.epochs)
    model, history = train_synthetic(cfg.synthetic_dir, hp, retrain=args.retrain)
    if history:
        plot_losses(history, cfg.synthetic_dir / "loss_curves.png", "Synthetic expanding-circle PINN")
    error = relative_l2_synthetic(model, hp)
    log.info("relative L2 error against the analytical circle: %.4f", error)
    plot_synthetic_fronts(
        synthetic_snapshots(model, hp),
        cfg.synthetic_dir / "front_snapshots.png",
        rate=hp.rate,
        radius=hp.radius,
    )
    log.info("figures written under %s", cfg.synthetic_dir)


def cmd_demo(args, cfg: ProjectConfig) -> None:
    out = cfg.root / "outputs" / "demo"
    setup_logging(out / "run.log")
    log.info("device: %s", get_device())
    run_demo(out, epochs=args.epochs)


def cmd_download(args, cfg: ProjectConfig, replace_log: bool = True) -> None:
    if replace_log:
        setup_logging(cfg.output_dir / "download.log")
    selected = {
        part.strip()
        for part in getattr(args, "components", "firms,landfire,gridmet").split(",")
        if part.strip()
    }
    if "firms" in selected and (args.rebuild or not cfg.firms_csv.exists()):
        download_firms(cfg, _firms_key(args.firms_key))
    elif "firms" in selected:
        log.info("keeping existing %s", cfg.firms_csv)
    if "landfire" in selected and (
        args.rebuild or any(not path.exists() for path in (cfg.elev_tif, cfg.slope_tif, cfg.fuel_tif))
    ):
        download_landfire(cfg)
    elif "landfire" in selected:
        log.info("keeping existing LANDFIRE rasters")
    if "gridmet" in selected and (args.rebuild or not cfg.gridmet_nc.exists()):
        download_gridmet(cfg)
    elif "gridmet" in selected:
        log.info("keeping existing %s", cfg.gridmet_nc)


def cmd_perimeters(args, cfg: ProjectConfig, replace_log: bool = True) -> None:
    if replace_log:
        setup_logging(cfg.output_dir / "perimeters.log")
    _require([cfg.firms_csv], "Download detections with: python main.py download --components firms")
    detections = _read_detections(cfg)
    build_perimeters(
        detections, cfg.perimeters_path,
        alpha=cfg.alpha, min_points=cfg.min_points, crs=cfg.target_crs,
    )


def cmd_stack(cfg: ProjectConfig, replace_log: bool = True) -> None:
    if replace_log:
        setup_logging(cfg.output_dir / "stack.log")
    build_stack(cfg)
    describe_stack(cfg.stack_nc)


def cmd_spread(cfg: ProjectConfig, replace_log: bool = True) -> None:
    if replace_log:
        setup_logging(cfg.output_dir / "spread.log")
    build_spread_rate(cfg)
    from wildfire_pinn.data_pipeline import load_dataset

    rate = load_dataset(cfg.rate_nc)
    day = 1 if rate.sizes.get("day", 1) > 1 else 0
    plot_scalar_grid(
        rate["x"].values, rate["y"].values, rate["R"].values[day],
        cfg.output_dir / "spread_rate.png",
        title=f"Spread rate on day index {day}",
        colorbar="R (m/s)",
        cmap="hot",
    )


def cmd_train(args, cfg: ProjectConfig, replace_log: bool = True):
    if replace_log:
        setup_logging(cfg.output_dir / "train.log")
    _require(
        list(cfg.model_inputs),
        "Build them with: python main.py pipeline",
    )
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    hp = RealConfig(epochs=args.epochs, k_train=cfg.k_train)
    model, history = train_real(
        case, cfg.output_dir, hp, retrain=args.retrain,
        split_scheme=cfg.split_scheme,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
    )
    if history:
        plot_losses(history, cfg.output_dir / "loss_curves.png", f"{cfg.name} level-set PINN")
    return model, case, hp


def _load_trained(cfg: ProjectConfig):
    """Checkpoint and fire case, same construction as `evaluate`. Does not train."""
    _require(
        list(cfg.model_inputs) + [cfg.output_dir / "levelset_pinn.pt"],
        "Train first with: python main.py train",
    )
    from wildfire_pinn.pinn import LevelSetPINN, load_checkpoint

    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    hp = RealConfig(k_train=cfg.k_train)
    model = LevelSetPINN(hp.hidden, hp.layers)
    load_checkpoint(cfg.output_dir / "levelset_pinn.pt", model, get_device())
    model.eval()
    return model, case, hp


def cmd_evaluate(cfg: ProjectConfig, model=None, case=None, hp: RealConfig | None = None) -> None:
    if model is None:
        setup_logging(cfg.output_dir / "evaluate.log")
        model, case, hp = _load_trained(cfg)
    from wildfire_pinn.data_pipeline import load_dataset

    stack = load_dataset(cfg.stack_nc)
    elevation = stack["elevation_m"].values if "elevation_m" in stack else None
    from wildfire_pinn.pinn import resolve_time_split

    time_split = resolve_time_split(
        len(case.perimeters), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if cfg.split_scheme == "chronological":
        boundary = f"train | validation (n={len(time_split.train)})"
    else:
        boundary = None
    plot_dashboard(
        case, len(time_split.train), cfg.output_dir / "dashboard.png",
        elevation=elevation, boundary_label=boundary,
    )
    rows = evaluate_real(
        model, case, hp.k_train,
        figure_path=cfg.output_dir / "heldout_fronts.png",
        geojson_path=cfg.output_dir / "fronts.geojson",
        case_name=cfg.name,
        split_scheme=cfg.split_scheme,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
    )
    write_metrics(cfg.output_dir / "metrics.json", rows)


def cmd_diagnostics(cfg: ProjectConfig) -> None:
    setup_logging(cfg.output_dir / "diagnostics.log")
    from wildfire_pinn.pinn import run_diagnostics

    model, case, hp = _load_trained(cfg)
    run_diagnostics(model, case, cfg.output_dir, hp.k_train, hp.c_wdir)


def cmd_playback(cfg: ProjectConfig) -> None:
    setup_logging(cfg.output_dir / "playback.log")
    from wildfire_pinn.pinn import write_playback

    model, case, _hp = _load_trained(cfg)
    write_playback(model, case, cfg.output_dir / "playback.gif")


def cmd_check(cfg: ProjectConfig) -> None:
    setup_logging(None)
    for path in cfg.raw_inputs + cfg.model_inputs:
        log.info("%s  %s", "ok  " if path.exists() else "miss", path)
    checkpoint = cfg.output_dir / "levelset_pinn.pt"
    log.info("%s  %s", "ok  " if checkpoint.exists() else "miss", checkpoint)
    if cfg.stack_nc.exists():
        describe_stack(cfg.stack_nc)


def cmd_pipeline(args, cfg: ProjectConfig) -> None:
    setup_logging(cfg.output_dir / "pipeline.log")
    log.info("device: %s", get_device())
    if not args.skip_download:
        cmd_download(args, cfg, replace_log=False)
    else:
        _require(list(cfg.raw_inputs), "Drop --skip-download, or place the raw files in data/.")
    if args.rebuild or not cfg.perimeters_path.exists():
        cmd_perimeters(args, cfg, replace_log=False)
    else:
        log.info("keeping existing %s", cfg.perimeters_path)
    if args.rebuild or not cfg.stack_nc.exists():
        cmd_stack(cfg, replace_log=False)
    else:
        log.info("keeping existing %s", cfg.stack_nc)
    if args.rebuild or not cfg.rate_nc.exists():
        cmd_spread(cfg, replace_log=False)
    else:
        log.info("keeping existing %s", cfg.rate_nc)
    model, case, hp = cmd_train(args, cfg, replace_log=False)
    cmd_evaluate(cfg, model=model, case=case, hp=hp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Level-set PINN for one wildfire. The default case is the Palisades Fire."
    )
    parser.add_argument("--root", type=Path, default=None, help="project directory (default: this repository)")
    parser.add_argument(
        "--name", default="palisades",
        help="fire dataset: palisades (default) or kincade. Other names are a custom window.",
    )
    parser.add_argument(
        "--bbox", default=None,
        help="west,south,east,north in degrees. Write --bbox=-118.7,34.0,-118.4,34.1 so the leading minus is not read as a flag",
    )
    parser.add_argument("--start", default=None, help="first day YYYY-MM-DD (default: 2025-01-07)")
    parser.add_argument("--end", default=None, help="last day YYYY-MM-DD (default: 2025-01-31)")
    parser.add_argument(
        "--crs", default=None,
        help="meter CRS for the model grid (default: EPSG:3310, California Albers)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("synthetic", help="train the analytical expanding-circle PINN")
    synthetic.add_argument("--epochs", type=int, default=2001)
    synthetic.add_argument("--retrain", action="store_true")

    demo = sub.add_parser("demo", help="train on a synthetic fire; no downloads or API key")
    demo.add_argument("--epochs", type=int, default=60)

    download = sub.add_parser("download", help="download FIRMS, LANDFIRE, and gridMET")
    download.add_argument("--firms-key", default=None)
    download.add_argument("--components", default="firms,landfire,gridmet")
    download.add_argument("--rebuild", action="store_true", help="download again even if files exist")

    perimeters = sub.add_parser("perimeters", help="build overpass hulls from firms_raw.csv")
    perimeters.add_argument("--rebuild", action="store_true")

    sub.add_parser("stack", help="reproject terrain, fuel, and weather onto one grid")
    sub.add_parser("spread-rate", help="write the first-order spread-rate field")

    train = sub.add_parser("train", help="train the Palisades level-set PINN")
    train.add_argument("--epochs", type=int, default=4000)
    train.add_argument("--retrain", action="store_true", help="ignore a saved checkpoint")

    sub.add_parser("evaluate", help="held-out front error, dashboard, and fronts.geojson")
    sub.add_parser("diagnostics", help="PDE residual maps and an eikonal check; no training")
    sub.add_parser("playback", help="GIF of observed hulls and predicted fronts at each overpass")
    sub.add_parser("check", help="show which pipeline files are present")

    pipeline = sub.add_parser("pipeline", help="download, build, train, and evaluate")
    pipeline.add_argument("--epochs", type=int, default=4000)
    pipeline.add_argument("--firms-key", default=None)
    pipeline.add_argument("--skip-download", action="store_true")
    pipeline.add_argument("--rebuild", action="store_true", help="recompute derived files and retrain")
    pipeline.add_argument("--retrain", action="store_true")
    return parser


def _iso_day(value: str, label: str) -> str:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise RuntimeError(f"{label} must be YYYY-MM-DD, got {value!r}")
    return value


def config_from_args(args) -> ProjectConfig:
    """Build the case from the command line.

    ``--name palisades`` and ``--name kincade`` fill the box, dates, and split.
    An explicit ``--bbox``, ``--start``, or ``--end`` overrides that dataset.
    """
    spec = DATASETS.get(args.name)
    kwargs = {
        "name": args.name,
        "split_scheme": spec.split_scheme if spec else "k_train",
        "train_fraction": spec.train_fraction if spec else 0.6,
        "val_fraction": spec.val_fraction if spec else 0.2,
    }
    if spec is not None:
        kwargs["bbox"] = spec.bbox
        kwargs["date_start"] = spec.date_start
        kwargs["date_end"] = spec.date_end
    if args.root is not None:
        kwargs["root"] = args.root
    if args.bbox:
        parts = [part.strip() for part in args.bbox.split(",")]
        if len(parts) != 4:
            raise RuntimeError("--bbox must be west,south,east,north in degrees")
        try:
            west, south, east, north = (float(part) for part in parts)
        except ValueError as exc:
            raise RuntimeError("--bbox must be four numbers: west,south,east,north") from exc
        if not (west < east and south < north):
            raise RuntimeError("--bbox must have west < east and south < north")
        if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
            raise RuntimeError("--bbox is outside longitude/latitude range")
        kwargs["bbox"] = (west, south, east, north)
    if args.start:
        kwargs["date_start"] = _iso_day(args.start, "--start")
    if args.end:
        kwargs["date_end"] = _iso_day(args.end, "--end")
    if kwargs.get("date_start", "2025-01-07") > kwargs.get("date_end", "2025-01-31"):
        raise RuntimeError("--start is after --end")
    if args.crs:
        kwargs["target_crs"] = args.crs
    return ProjectConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    commands = {
        "synthetic": lambda: cmd_synthetic(args, cfg),
        "demo": lambda: cmd_demo(args, cfg),
        "download": lambda: cmd_download(args, cfg),
        "perimeters": lambda: cmd_perimeters(args, cfg),
        "stack": lambda: cmd_stack(cfg),
        "spread-rate": lambda: cmd_spread(cfg),
        "train": lambda: cmd_train(args, cfg),
        "evaluate": lambda: cmd_evaluate(cfg),
        "diagnostics": lambda: cmd_diagnostics(cfg),
        "playback": lambda: cmd_playback(cfg),
        "check": lambda: cmd_check(cfg),
        "pipeline": lambda: cmd_pipeline(args, cfg),
    }
    try:
        commands[args.command]()
    except RuntimeError as exc:
        setup_logging(None)
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
