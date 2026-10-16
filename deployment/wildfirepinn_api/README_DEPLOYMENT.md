# Local research API

Experiment 5 adds a Vertex `/predict` adapter, a teammate client, and the cloud checklist. Container and remote deployment were not executed in that session: Docker and gcloud are not installed on the authoring machine. See `EXPERIMENT_5_RESULTS.md` and `CLOUD_DEPLOYMENT_CHECKLIST.md`. The model choice below is unchanged.

Experimental prototype. Not an operational wildfire forecast. The default model is `ray_inverse_solver`, a numerical hybrid that estimates radial speeds and advances the Godunov level-set solver. It is not a PINN.

The PINN endpoint is `direct_pinn_experimental`. It answers only the Experiment 2A.3 F1 seed-0 circle. Details and the synthetic scores are in `EXPERIMENT_4_RESULTS.md` and `DEPLOYMENT_READINESS.md`. Component reuse is in `ARCHITECTURE.md`.

## Start

From the repository root, with the existing Python environment (PyTorch, NumPy, SciPy, Shapely, scikit-image, FastAPI):

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
python -m uvicorn deployment.wildfirepinn_api.service.app:app --host 127.0.0.1 --port 8080
```

The key must be at least 16 characters. It is an environment variable. It is not written into the repository. Stop the server with Ctrl+C.

A new environment can install the CPU wheel and then the API requirements:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r deployment/wildfirepinn_api/requirements.txt
```

## Test

In a second shell, from the repository root:

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
python -m unittest deployment.wildfirepinn_api.tests.test_estimators deployment.wildfirepinn_api.tests.test_api deployment.wildfirepinn_api.tests.test_pinn
python -m deployment.wildfirepinn_api.tests.smoke_live
python -m deployment.wildfirepinn_api.examples.client
```

`smoke_live` calls `/health`, `/models`, and `/forecast`, checks the polygon and the 600 s horizon, and checks that a wrong key is rejected. The example client posts `examples/synthetic_circle_request.json` (the same circle: observations at 0, 200, and 400 s, forecast at 600 s, local meters).

```powershell
$headers = @{ "X-API-Key" = $env:WILDFIREPINN_API_KEY }
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/models -Headers $headers
Invoke-RestMethod http://127.0.0.1:8080/version
```

`/health` and `/version` do not require the key. `/models` and `/forecast` do. A missing or wrong key returns 401. Interactive docs stay off unless `WILDFIREPINN_ENABLE_DOCS=1`.

The request schema is `examples/forecast_request.schema.json`. Coordinates are meters. `EPSG:4326` is rejected. `burned_area_m2` is the grid mask. `perimeter_polygon_area_m2` is the contour polygon. `uncertainty` is null.

## Container

Build from the repository root:

```powershell
docker build -f deployment/wildfirepinn_api/Dockerfile -t wildfirepinn-research-api:local .
docker run --rm -p 127.0.0.1:8080:8080 -e WILDFIREPINN_API_KEY=replace-with-a-long-random-string wildfirepinn-research-api:local
```

The process listens on 0.0.0.0 inside the container, on `AIP_HTTP_PORT` if Vertex set it, otherwise on `PORT` (default 8080). In the default `api_key` mode it exits unless `WILDFIREPINN_API_KEY` has at least 16 characters. With `WILDFIREPINN_PREDICT_AUTH=vertex_iam` it starts without that variable: `POST /predict` accepts `{"instances": [...]}` and returns `{"predictions": [...]}`, while `GET /models` and `POST /forecast` return 401 until a key of at least 16 characters is also set. Use `vertex_iam` only on a Vertex endpoint that requires IAM. The image does not train and does not download a dataset. Image tag for this release is `wildfirepinn-research-api:0.1.1`.

The Docker CLI was not installed on the machine that added this service, so the image was not built or run here. The Dockerfile is the build definition. A machine with Docker should use the commands above.

Cloud Run commands, and the reason Vertex AI is not the first target, are in `DEPLOYMENT_READINESS.md`. Those commands have not been run.
