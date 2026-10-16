"""Example call. The default model is the ray hybrid, not the PINN.

    $env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
    python -m deployment.wildfirepinn_api.examples.client
"""

from __future__ import annotations

import json

from deployment.wildfirepinn_api.client import forecast, synthetic_circle


def example_request() -> dict:
    """Same circle the unit tests use: 0.05 m/s growth, forecast 200 s past the last ring."""
    from deployment.wildfirepinn_api.client import build_forecast_body

    return build_forecast_body(
        [synthetic_circle(80.0), synthetic_circle(90.0), synthetic_circle(100.0)],
        [0.0, 200.0, 400.0],
        200.0,
    )


def main() -> None:
    body = forecast(
        [synthetic_circle(80.0), synthetic_circle(90.0), synthetic_circle(100.0)],
        [0.0, 200.0, 400.0],
        200.0,
    )
    print(json.dumps({
        "model_id": body["model_id"],
        "model_version": body["model_version"],
        "forecast_time_s": body["forecast_time_s"],
        "units": body["units"],
        "burned_area_m2": body["burned_area_m2"],
        "perimeter_type": body["perimeter"]["type"],
        "operational_wildfire_service": body["operational_wildfire_service"],
        "warnings": body["warnings"][:2],
    }, indent=2))


if __name__ == "__main__":
    main()
