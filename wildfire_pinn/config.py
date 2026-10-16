"""Paths and experiment settings for one wildfire case.

The default case is the Palisades Fire. Another fire is the same pipeline
with a different name, bounding box, and date window. Files for that name
stay beside the Palisades files; they do not overwrite them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# West, south, east, north in WGS84 degrees.
# Palisades window is the one used in the Colab notebook.
PALISADES_BBOX = (-118.70, 34.00, -118.42, 34.14)
# Kincade started the night of 23 October 2019 at John Kincade Road
# (38.792458 N, 122.780053 W) and was contained on 6 November. The box
# covers that ignition and the run west and south toward Healdsburg and Windsor.
KINCADE_BBOX = (-123.12, 38.45, -122.60, 38.92)
_CASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")


@dataclass(frozen=True)
class DatasetSpec:
    """One fire event: where it was, when it burned, and how overpasses are split.

    ``k_train`` keeps the Palisades rule: the first ``k_train`` overpasses are
    training data and the rest are held out. ``chronological`` cuts the
    time-ordered overpasses into train, then validation, then test. Neither
    scheme shuffles.
    """

    name: str
    bbox: tuple[float, float, float, float]
    date_start: str
    date_end: str
    split_scheme: str
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "palisades": DatasetSpec(
        name="palisades",
        bbox=PALISADES_BBOX,
        date_start="2025-01-07",
        date_end="2025-01-31",
        split_scheme="k_train",
        description="January 2025 Palisades Fire. First k_train overpasses train; the rest are held out.",
    ),
    "kincade": DatasetSpec(
        name="kincade",
        bbox=KINCADE_BBOX,
        date_start="2019-10-23",
        date_end="2019-11-06",
        split_scheme="chronological",
        train_fraction=0.6,
        val_fraction=0.2,
        description=(
            "2019 Kincade Fire in Sonoma County, from ignition through containment. "
            "Overpasses stay in time order and are cut 60% train, 20% validation, 20% test."
        ),
    ),
}


@dataclass
class ProjectConfig:
    """Filesystem layout and the geospatial window for one case study.

    ``target_crs`` is the meter grid the PINN trains on. EPSG:3310 is
    California Albers, which is the right default inside California and the
    wrong one outside it.
    """

    root: Path = field(default_factory=lambda: PROJECT_ROOT)
    name: str = "palisades"
    bbox: tuple[float, float, float, float] = PALISADES_BBOX
    date_start: str = "2025-01-07"
    date_end: str = "2025-01-31"
    target_crs: str = "EPSG:3310"
    resolution_m: float = 90.0
    firms_product: str = "VIIRS_SNPP_SP"
    # Alpha is in the units of the detection coordinates (degrees).
    # 100 was chosen from the notebook's alpha sweep; it is not a meter radius.
    alpha: float = 100.0
    min_points: int = 4
    k_train: int = 10
    # k_train reproduces Palisades. chronological is a prefix train/validation/test cut.
    split_scheme: str = "k_train"
    train_fraction: float = 0.6
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not _CASE_NAME.match(self.name):
            raise RuntimeError(
                f"Case name {self.name!r} must be letters, numbers, '_' or '-', "
                "so it can be used as a filename."
            )

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs" / self.name

    @property
    def synthetic_dir(self) -> Path:
        return self.root / "outputs" / "synthetic"

    @property
    def firms_csv(self) -> Path:
        # The Palisades download already lives at this notebook filename.
        if self.name == "palisades":
            return self.data_dir / "firms_raw.csv"
        return self.data_dir / f"{self.name}_firms.csv"

    @property
    def perimeters_path(self) -> Path:
        return self.data_dir / f"{self.name}_perimeters.geojson"

    @property
    def elev_tif(self) -> Path:
        return self.data_dir / f"{self.name}_elev.tif"

    @property
    def slope_tif(self) -> Path:
        return self.data_dir / f"{self.name}_sloped.tif"

    @property
    def fuel_tif(self) -> Path:
        return self.data_dir / f"{self.name}_fbfm40.tif"

    @property
    def gridmet_nc(self) -> Path:
        # Clipped to this case's bbox, so a second fire must not reuse the Palisades file.
        if self.name == "palisades":
            return self.data_dir / "gridmet_jan2025.nc"
        return self.data_dir / f"{self.name}_gridmet.nc"

    @property
    def stack_nc(self) -> Path:
        return self.data_dir / f"{self.name}_stack.nc"

    @property
    def rate_nc(self) -> Path:
        return self.data_dir / f"{self.name}_R.nc"

    @property
    def raw_inputs(self) -> tuple[Path, ...]:
        return (self.firms_csv, self.elev_tif, self.slope_tif, self.fuel_tif, self.gridmet_nc)

    @property
    def model_inputs(self) -> tuple[Path, ...]:
        return (self.stack_nc, self.rate_nc, self.perimeters_path)
