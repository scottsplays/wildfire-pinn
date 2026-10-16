"""Hard bounds for one research forecast. These are not physical fire limits."""

MAX_BODY_BYTES = 1_000_000
MAX_OBSERVATIONS = 4
MAX_VERTICES = 4_000
MAX_CELLS_PER_AXIS = 201
MIN_CELLS_PER_AXIS = 3
MIN_RESOLUTION_M = 10.0
MAX_RESOLUTION_M = 200.0
MAX_DOMAIN_SIDE_M = 8_000.0
MAX_FORECAST_TIME_S = 20_000.0
MAX_HORIZON_S = 8_000.0
FORECAST_TIMEOUT_S = 60.0
MIN_API_KEY_LENGTH = 16
MAX_RATE_M_S = 0.15
MIN_RATE_M_S = 1.0e-4

GEOGRAPHIC_CRS = frozenset({
    "EPSG:4326",
    "EPSG:4269",
    "EPSG:4258",
    "EPSG:4267",
    "EPSG:4979",
})

FOURIER_WAVELENGTH_M = 1500.0
PINN_BENCHMARK = "experiment2a3_f1_spatial_product"
DISCLAIMER = (
    "Experimental research prototype. Not an operational wildfire emergency forecasting service."
)
