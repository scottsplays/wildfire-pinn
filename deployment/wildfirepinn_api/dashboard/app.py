"""Local map for the research forecast API.

Plots come from POST /forecast. This file does not run the solver.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

API_URL = os.environ.get("WILDFIREPINN_API_URL", "http://127.0.0.1:8080").rstrip("/")
FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_circle_request.json"
PINN_BENCHMARK = "experiment2a3_f1_spatial_product"
PINN_MODEL = "direct_pinn_experimental"

st.set_page_config(page_title="WildfirePINN research dashboard", layout="wide")


def _key() -> str:
    return os.environ.get("WILDFIREPINN_API_KEY", "")


def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    key = _key()
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{API_URL}{path}", data=data, method=method)
    if payload is not None:
        request.add_header("content-type", "application/json")
    if key:
        request.add_header("x-api-key", key)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"detail": raw}
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, {"detail": f"Could not reach {API_URL}: {exc.reason}"}


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def forecast_body(fixture: dict, model: str, horizon_s: float) -> dict:
    """Build a /forecast body from the synthetic circle fixture."""
    if model == PINN_MODEL:
        return {
            "model": PINN_MODEL,
            "coordinate_system": "local_cartesian_m",
            "supported_benchmark": PINN_BENCHMARK,
            "forecast_time_s": float(horizon_s),
        }
    last = float(fixture["observations"][-1]["time_s"])
    body = {
        "model": model,
        "coordinate_system": fixture["coordinate_system"],
        "grid": fixture["grid"],
        "observations": fixture["observations"],
        "forecast_time_s": last + float(horizon_s),
    }
    if model == "fourier_inverse_solver":
        body["rate_parameters"] = {"family": "sin_kx_cos_ky", "wavelength_m": 1500}
    return body


def _rings(geometry: dict):
    if geometry.get("type") == "Polygon":
        yield geometry["coordinates"][0]
    elif geometry.get("type") == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield polygon[0]


def draw_map(request_body: dict, response: dict) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    observations = request_body.get("observations") or []
    colors = ["#9ecae1", "#6baed6", "#2171b5", "#08306b"]
    all_x: list[float] = []
    all_y: list[float] = []
    for index, observation in enumerate(observations):
        ring = observation["perimeter"]["coordinates"][0]
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        all_x.extend(xs)
        all_y.extend(ys)
        axis.plot(
            xs,
            ys,
            color=colors[index % len(colors)],
            linewidth=1.8,
            label=f"observed t = {observation['time_s']:.0f} s",
        )
    forecast_label = f"forecast t = {response['forecast_time_s']:.0f} s"
    for index, ring in enumerate(_rings(response["perimeter"])):
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        all_x.extend(xs)
        all_y.extend(ys)
        axis.plot(xs, ys, color="#cb181d", linewidth=2.2, label=forecast_label if index == 0 else None)
    if all_x and all_y:
        span = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
        pad = max(40.0, 0.15 * span)
        axis.set_xlim(min(all_x) - pad, max(all_x) + pad)
        axis.set_ylim(min(all_y) - pad, max(all_y) + pad)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Local Cartesian meters. These axes are not longitude and latitude.")
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    st.title("WildfirePINN research dashboard")
    st.caption(
        "Experimental research prototype. Not an operational wildfire emergency forecasting service. "
        "The default ray model is a numerical hybrid, not a PINN."
    )

    if len(_key()) < 16:
        st.error("Set WILDFIREPINN_API_KEY to at least 16 characters in this terminal before starting Streamlit.")
        st.stop()

    status, health = _request("GET", "/health")
    if status != 200:
        st.error(health.get("detail") or f"Health check failed ({status}). Start the API at {API_URL}.")
        st.stop()

    status, registry = _request("GET", "/models")
    if status != 200:
        detail = registry.get("detail") or registry
        st.error(f"GET /models failed ({status}): {detail}")
        st.stop()

    models = registry["models"]
    available = [name for name, record in models.items() if record.get("available", True)]
    default = registry.get("default_model")
    if default not in available:
        default = available[0]

    fixture = load_fixture()
    last_observation_s = float(fixture["observations"][-1]["time_s"])

    st.subheader("Models reported by the API")
    st.dataframe(
        [
            {
                "model": name,
                "version": record.get("version"),
                "available": record.get("available", True),
                "PDE-residual PINN": record.get("genuine_pde_residual_pinn"),
                "default": name == default,
            }
            for name, record in models.items()
        ],
        hide_index=True,
        width="stretch",
    )

    with st.sidebar:
        st.header("Forecast")
        st.write(f"API: `{API_URL}`")
        model = st.selectbox(
            "Model",
            available,
            index=available.index(default),
            format_func=lambda name: f"{name}{' (PINN benchmark)' if name == PINN_MODEL else ''}",
        )
        record = models[model]
        if model == PINN_MODEL:
            horizon = st.slider("Forecast time from ignition (s)", min_value=0, max_value=4000, value=4000, step=100)
            st.caption("This checkpoint does not accept the synthetic circle perimeters.")
        else:
            horizon = st.slider(
                "Forecast horizon after the last observed perimeter (s)",
                min_value=50,
                max_value=2000,
                value=200,
                step=50,
            )
            st.caption(f"Last observation is {last_observation_s:.0f} s. Requested forecast time is {last_observation_s + horizon:.0f} s.")
        st.write(record.get("forecast_method") or "")
        request_forecast = st.button("Request forecast", type="primary")

    st.subheader("Synthetic scenario")
    st.write(
        "Loaded from `deployment/wildfirepinn_api/examples/synthetic_circle_request.json`. "
        "Three local-meter circles, radii 80 m, 90 m, and 100 m, at 0 s, 200 s, and 400 s. "
        "The grid is [-400, 400] m at 20 m resolution."
    )
    for item in record.get("limitations") or []:
        st.warning(item)

    signature = (model, float(horizon))
    if request_forecast:
        body = forecast_body(fixture, model, float(horizon))
        code, response = _request("POST", "/forecast", body)
        if code != 200 or "perimeter" not in response:
            detail = response.get("detail") or response
            st.error(f"POST /forecast failed ({code}): {detail}")
            st.session_state.pop("forecast", None)
        else:
            st.session_state["forecast"] = {"signature": signature, "request": body, "response": response}

    saved = st.session_state.get("forecast")
    if not saved or saved["signature"] != signature:
        st.info("Choose a model and horizon, then request a forecast. The map stays empty until the API responds.")
        return

    response = saved["response"]
    request_body = saved["request"]
    horizon_s = response["forecast_time_s"] - last_observation_s if request_body.get("observations") else None

    left, right = st.columns(2)
    with left:
        st.metric("Forecast time", f"{response['forecast_time_s']:.0f} s")
        if horizon_s is None:
            st.metric("Horizon after last observation", "not used")
        else:
            st.metric("Horizon after last observation", f"{horizon_s:.0f} s")
    with right:
        st.metric("Burned area (grid mask)", f"{response['burned_area_m2']:.0f} m²")
        st.metric("Perimeter polygon area", f"{response['perimeter_polygon_area_m2']:.0f} m²")

    st.caption(
        f"{response['model_id']} {response['model_version']} · "
        f"{response['coordinate_system']} · "
        f"inference {response['elapsed_inference_s']:.3f} s · "
        "uncertainty is null"
    )
    figure = draw_map(request_body, response)
    st.pyplot(figure, clear_figure=True)
    for warning in response.get("warnings") or []:
        st.caption(warning)
    with st.expander("API response"):
        public = {key: value for key, value in response.items() if key != "perimeter"}
        public["perimeter_type"] = response["perimeter"]["type"]
        public["perimeter_points"] = len(next(_rings(response["perimeter"])))
        st.json(public)


main()
