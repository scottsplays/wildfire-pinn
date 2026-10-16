# Deployment readiness

Experiment 5 moved the cloud procedure to `CLOUD_DEPLOYMENT_CHECKLIST.md`. The model table below is the Experiment 4 selection and is unchanged. No remote deployment has been performed.

This service has been run locally with `uvicorn` on 127.0.0.1:8080. The live smoke test called `/health`, `/models`, and `/forecast`, checked the polygon and the 600 s horizon, and rejected a wrong API key. It has not been deployed. No Google Cloud resource was created, and no remote request was sent.

The Docker CLI was not available in that session, so `docker build` was not executed. The Dockerfile is in this directory.

## Model choice

| Model | PDE-residual PINN | Physics-constrained inverse | Needs a prescribed R | Infers R from perimeters | Held-out synthetic test | Real-fire validation | On the API | General forecasting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_rate_solver | no | no | optional scalar | one global speed | yes, weaker than ray on all six fires | no | yes | no |
| ray_inverse_solver | no | yes, through the Godunov solver | no | radial speeds only | yes, best when the formula is unknown | no | yes, default | no |
| fourier_inverse_solver | no | yes, through the Godunov solver | yes, 1500 m product family | two coefficients inside that family | yes on case A; fails on B and C | no | yes, specialized | no |
| direct_pinn_experimental | yes, F1 seed 0 | no | the training field is built into the checkpoint | no | benchmark only; IoU 0.962 at 4000 s in Experiment 2A.3 | no | yes, benchmark id required | no |
| Experiment 3 rate MLP | no | a level-set adjoint was used in training | no | attempted | no; wrong growth-gap sign on three seeds | no | not served | no |

The default is `ray_inverse_solver`. It is a numerical hybrid: observed radial speeds painted onto a grid, then the isotropic level-set solver. It is not a PINN. The Fourier option stays available only when the caller explicitly sends `family = sin_kx_cos_ky` and `wavelength_m = 1500`. The F1 PINN stays available only with `supported_benchmark = experiment2a3_f1_spatial_product`.

## Cloud Run rather than Vertex AI

The workload is one CPU forecast per HTTP call. Ray inference is a fraction of a second. Fourier fitting is a few seconds. The PINN evaluation is one forward pass. There is no training job, no batch prediction table, and no model-monitoring loop.

Cloud Run runs this container as an HTTP service. Vertex AI custom prediction would put the same container behind a different prediction route and would not make the solver more accurate. Cloud Run is the simpler fit. Vertex is not configured.

## What a later deploy would do

These commands are the procedure. They were not run.

1. Create an Artifact Registry Docker repository in the region the team chooses, co-located with the Cloud Run service.
2. Build and push `deployment/wildfirepinn_api/Dockerfile` from the repository root.
3. Store the API key in Secret Manager. Mount it as `WILDFIREPINN_API_KEY`. Do not put the key in the image or the service YAML in git.
4. Deploy to Cloud Run with request-based billing, minimum instances 0, 1 vCPU, 4 GiB memory, request timeout 60 seconds, and container concurrency 1. The solver is not a multi-request batch worker, and the image includes PyTorch.
5. Require Cloud Run authenticated invocations (IAM) in addition to the application API key. `/health` stays unauthenticated inside the container so the platform probe can call it; IAM still sits in front of the URL if invocations are authenticated.
6. Send one synthetic forecast from outside the laptop and check the returned horizon and polygon before telling the team the URL is live.

```text
gcloud artifacts repositories create wildfirepinn --repository-format=docker --location=REGION
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1
gcloud run deploy wildfirepinn-research --image REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1 --region=REGION --no-allow-unauthenticated --memory=4Gi --cpu=1 --timeout=60 --concurrency=1 --min-instances=0 --set-secrets=WILDFIREPINN_API_KEY=wildfirepinn-api-key:latest
```

Replace `PROJECT` and `REGION`. Create the secret first. Do not run this until someone accepts the bill.

## Cost sketch

Published Cloud Run request-based list prices, retrieved 11 October 2026 from the Cloud Run pricing page, are $0.000024 per vCPU-second, $0.0000025 per GiB-second, and $0.40 per million requests, with a monthly free tier of 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million requests. Occasional research calls at 1 vCPU and 4 GiB stay inside that free tier.

Artifact Registry storage above 0.5 GiB-month is listed at $0.000136986 per GiB-hour. A multi-gigabyte torch image is on the order of a dollar per month if it is left in the registry. Cloud Build minutes for the image are separate. This is not a quote, and no charge was created here.

## Still true after the API exists

- The ray model extrapolates measured speeds into cells the front has not crossed.
- The Fourier model knows its functional family.
- The PINN does not accept a real perimeter.
- Geographic CRS values are rejected. Projected meters are not reprojected or checked against a fire.
- `uncertainty` is null. No calibrated interval is invented.
- `real_fire.evaluation_status` returns `not_executed` and `metrics: null`.
