# Team handoff

This is a research API. It is not an operational wildfire forecast. The default model, `ray_inverse_solver`, is a numerical hybrid: it estimates radial speeds from the perimeters you send and advances the isotropic Godunov level-set solver. It is not a PINN.

`direct_pinn_experimental` is a PDE-residual-trained Fourier PINN. It answers only the Experiment 2A.3 circle. Use `forecast_benchmark`, not `forecast`, for that call.

## Call

From the repository root, with the server already running:

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
$env:WILDFIREPINN_API_URL = "http://127.0.0.1:8080"
python -c "from deployment.wildfirepinn_api.client import forecast, synthetic_circle; body = forecast([synthetic_circle(80), synthetic_circle(90), synthetic_circle(100)], [0, 200, 400], 200); print(body['model_id'], body['forecast_time_s'], body['burned_area_m2'])"
```

That circle grows at 0.05 m/s. Observations are at 0, 200, and 400 s. The horizon is 200 s, so the forecast time is 600 s. Coordinates are local meters, not longitude and latitude.

```python
from deployment.wildfirepinn_api.client import forecast, forecast_benchmark, synthetic_circle

result = forecast(
    observed_perimeters=[synthetic_circle(80), synthetic_circle(90), synthetic_circle(100)],
    timestamps=[0, 200, 400],
    forecast_horizon=200,
    model="ray_inverse_solver",
)
# result["perimeter"] is a GeoJSON Polygon in meters
# result["burned_area_m2"] counts grid cells with phi < 0
# result["perimeter_polygon_area_m2"] is the contour polygon
# result["uncertainty"] is None
# result["operational_wildfire_service"] is False

benchmark = forecast_benchmark(4000)
```

The key is read from `WILDFIREPINN_API_KEY`. Do not put it in source code. A failed request raises `ForecastClientError` with `status_code` and `detail`.

`forecast_horizon` is seconds after the last timestamp. The client builds `forecast_time_s` and a padded 20 m grid. Pass `grid=` when the fire may leave that pad. The pad is 400 m.

## Which model accepts what

| Model | What it is | Perimeters | Extra input |
| --- | --- | --- | --- |
| ray_inverse_solver | numerical hybrid, the default | at least two | none |
| constant_rate_solver | one speed in the Godunov solver | two, or one if `rate_m_s` is set | optional `{"family": "constant", "rate_m_s": 0.05}` |
| fourier_inverse_solver | numerical hybrid for one prescribed family | at least two | `{"family": "sin_kx_cos_ky", "wavelength_m": 1500}` |
| direct_pinn_experimental | PDE-residual PINN, F1 seed 0 | none | `forecast_benchmark(time_s)` with time in [0, 4000] |

Projected coordinates must already be meters, for example `coordinate_system="projected_meters", crs="EPSG:3310"`. `EPSG:4326` is rejected. The service does not reproject.

A real-fire score has not been run. `deployment.wildfirepinn_api.real_fire.evaluation_status()` returns `metrics: null`.
