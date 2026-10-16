# Architecture for the research forecast service

This note records what already existed and how the local API reuses it. No earlier experiment, checkpoint, or solver file was rewritten.

## Validated numerical solver

`experiments/wildfirepinn_v2_diagnostics/solver.py` is the isotropic Godunov / Osher–Sethian integrator

```text
phi_t + R(x, y) |grad phi| = 0
```

with phi negative inside the burned region and R in m/s. The API calls `propagate` and does not copy the scheme. A normal-dependent rate R(x, y, n) is a different Hamiltonian. `directional.py` implements one wind-advection split for a different experiment and is not used here. Anisotropic rate fields are unsupported.

## Hard-initial-condition PINN

The checkpoint served as `direct_pinn_experimental` is Experiment 2A.3 F1, seed 0:

`experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/runs/F1_seed0/levelset_pinn.pt`

It is a Fourier-feature network with the hard initial condition

```text
Phi = phi0 / L + (t / T) g_theta
```

trained with the level-set residual on the known product rate. The adapter in `pinn_adapter.py` loads that file with `load_checkpoint`, uses L = 4000 m and T = 4000 s, and evaluates only the training circle: domain [-2000, 2000] m, ignition (200, -150) m, radius 400 m, grid n = 201. Experiment 1B constant-rate checkpoints stay on disk. They are not substituted when a request falls outside this benchmark.

## Experiment 3 inverse estimators

`experiments/wildfirepinn_v3/experiment3/inverse.py` supplies the three numerical fits:

- `constant_rate`: one speed from the change in mean contour radius.
- `ray_rate`: radial speeds in 180 angle bins, painted onto the grid and extrapolated outside the swept radii.
- `fit_parametric`: Nelder–Mead on R = r0 (1 + a sin(kx) cos(ky)) at the frozen 1500 m wavelength and the frozen start list in `configs.json`.

`interface.py` calls those functions. It does not import the hidden true rate. The kept Fourier start is the one with the smaller training-contour error. Forecast times are not put in the fitting view.

## Geometry

`wildfire_pinn/geometry.py` provides `signed_distance` (negative inside, meters) and `zero_contours`. The API rasterizes a GeoJSON polygon with the first and extracts the forecast polygon with the second. Grid cell area and polygon area are returned as different fields. Coordinates stay in the meters the caller sent. EPSG:4326 and the other listed geographic codes are rejected. The service does not reproject.

## What the service adds

`deployment/wildfirepinn_api/` is a FastAPI shell around those pieces: a shared forecast result, request limits, an API key from the environment, and four model ids. Startup does not download data and does not train. The MLP inverse from Experiment 3 is not served. Its forecast did not recover the growth asymmetry.

## What is not claimed

The ray model is the default because it is the inverse method that can run without being told the true formula. It is a numerical hybrid, not a PINN. The F1 network is a PDE-residual PINN and is callable only on its training benchmark. Neither has a real-fire holdout.
