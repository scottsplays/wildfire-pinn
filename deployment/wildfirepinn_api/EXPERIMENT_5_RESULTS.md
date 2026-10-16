# Experiment 5 results

Deployment preparation for the Experiment 4 research API. Forecasting code was not retrained and was not changed. Experiment 4 results were not edited. The Gauss-inspired heat model was not started. No Google Cloud resource was created, and no remote request was sent.

## What was verified locally

The service still loads the F1 seed-0 checkpoint (57,000 bytes) at startup and does not read a training dataset or fit a rate while starting. `GET /models` marks `direct_pinn_experimental` unavailable when that load fails, and `POST /forecast` for that model then returns 409. The ray, constant, and Fourier paths do not need the checkpoint.

Requirements are pinned to the versions that passed the local tests, including `torch==2.12.1` from the CPU index. The Dockerfile copies the solver, the Experiment 3 inverse modules, the F1 checkpoint, and the API package. It listens on `AIP_HTTP_PORT` or `PORT`, binds `0.0.0.0` only inside the image, and exits when `WILDFIREPINN_API_KEY` is missing or shorter than 16 characters. `.env.example` still contains the placeholder string. No API key file is in the package.

`POST /predict` accepts the Vertex envelope `{"instances": [forecast body]}` and returns `{"predictions": [...]}`. `/forecast` is unchanged and still requires the API key. With `WILDFIREPINN_PREDICT_AUTH=vertex_iam`, `/predict` skips the application key and `/forecast` still returns 401 without it.

Unit tests: 21 passed, including the Experiment 4 API tests, the PINN benchmark comparison, the Vertex envelope, the disabled-PINN path, and the teammate client.

A local `uvicorn` process on 127.0.0.1:8080 then passed `python -m deployment.wildfirepinn_api.tests.smoke_live`:

- `GET /health` returned ok.
- `GET /models` listed the ray hybrid as the default.
- `GET /version` reported `cloud_deployment: not_deployed`.
- `POST /forecast` for `ray_inverse_solver` returned a valid polygon, forecast time 600 s, meters, and burned area 27,600 m².
- `POST /predict` returned the same model id inside `predictions`.
- `direct_pinn_experimental` on `experiment2a3_f1_spatial_product` at 4000 s returned version `experiment2a3-f1-seed0` and grid shape `[201, 201]`.
- `EPSG:4326` returned 422.
- A 20,000 s forecast returned 422.
- A wrong API key returned 401.

The example client printed `ray_inverse_solver`, version `experiment3-ray-v1`, and `operational_wildfire_service: false`. Application logs did not contain the API key. The ray response text calls that model a numerical hybrid.

## What failed

Docker validation did not run. `docker` is not on `PATH`, and `C:\Program Files\Docker\Docker\resources\bin\docker.exe` is absent. gcloud is also absent, so nothing was built, pushed, or deployed.

Because of that, these definition-of-done items are not met: container build, container integration test, an authenticated cloud request, a remote forecast, and a remote PINN call.

## Cloud choice

Cloud Run is ready as a written procedure and is the recommended target for an authenticated HTTP API. Vertex AI is not ready to receive this image unchanged under `endpoint.predict()`. The `/predict` adapter is implemented and was tested through the local TestClient and the local smoke test. It has not been deployed to a Vertex endpoint. Vertex would also keep at least one `n1-standard-4` replica running, which Cloud Run with minimum instances 0 does not.

## Did a remote deployment occur

No.

## Next command

Install Docker Desktop, then from the repository root:

```text
docker build -f deployment/wildfirepinn_api/Dockerfile -t wildfirepinn-research-api:0.1.0 .
```

Smoke-test that container with the commands in `CLOUD_DEPLOYMENT_CHECKLIST.md` before any `gcloud` command. Push and Cloud Run deploy stay blocked until someone approves the bill.
