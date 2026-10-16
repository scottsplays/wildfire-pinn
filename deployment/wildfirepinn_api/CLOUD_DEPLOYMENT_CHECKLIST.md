# Cloud deployment checklist

No cloud resource has been created. Do not run the `gcloud` commands until someone accepts the bill. The API must not be deployed with unauthenticated access.

Image tag: `wildfirepinn-research-api:0.1.1`.

Replace `PROJECT` and `REGION` (a Tier 1 region such as `us-central1` is the cheaper list price). Pick names the team controls.

## Recommendation

Use Cloud Run when teammates need an authenticated HTTP API. That matches `/health`, `/models`, `/version`, and `/forecast`.

Use Vertex AI when the team needs a managed prediction endpoint or the Vertex model registry. `POST /predict` accepts `{"instances": [...]}` and returns `{"predictions": [...]}`. Set `WILDFIREPINN_PREDICT_AUTH=vertex_iam` and require Vertex IAM on the endpoint. Image 0.1.1 starts in that mode without an application key. `GET /models` and `POST /forecast` return 401 until a key of at least 16 characters is configured. Keep `vertex_iam` off any service that allows public invocations.

## One-time project setup

```text
gcloud config set project PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create wildfirepinn --repository-format=docker --location=REGION
```

Create the API key outside the repository and store it in Secret Manager. Do not pass it on a command line that is logged, and do not commit it.

```text
gcloud secrets create wildfirepinn-api-key --replication-policy=automatic
gcloud secrets versions add wildfirepinn-api-key --data-file=KEYFILE
```

Delete `KEYFILE` after the secret version exists.

Runtime service account, with access only to that secret:

```text
gcloud iam service-accounts create wildfirepinn-runtime --display-name="WildfirePINN research API"
gcloud secrets add-iam-policy-binding wildfirepinn-api-key --member=serviceAccount:wildfirepinn-runtime@PROJECT.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
```

The deploy identity needs `roles/run.admin`, `roles/iam.serviceAccountUser` on that runtime account, and `roles/artifactregistry.writer`. It does not need project Owner.

## Build and Cloud Run

Build on a machine that has Docker, from the repository root, and smoke-test before any push:

```text
docker build -f deployment/wildfirepinn_api/Dockerfile -t wildfirepinn-research-api:0.1.1 .
docker run --rm -p 127.0.0.1:8080:8080 -e WILDFIREPINN_API_KEY=LOCAL_TEST_KEY_16CHARS wildfirepinn-research-api:0.1.1
```

In another shell, with the same key:

```text
set WILDFIREPINN_API_KEY=LOCAL_TEST_KEY_16CHARS
python -m deployment.wildfirepinn_api.tests.smoke_live
```

Push and deploy only after that smoke test passes:

```text
docker tag wildfirepinn-research-api:0.1.1 REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1
docker push REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1
gcloud run deploy wildfirepinn-research --image REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1 --region=REGION --no-allow-unauthenticated --service-account=wildfirepinn-runtime@PROJECT.iam.gserviceaccount.com --memory=4Gi --cpu=1 --timeout=60 --concurrency=1 --min-instances=0 --max-instances=2 --set-secrets=WILDFIREPINN_API_KEY=wildfirepinn-api-key:latest
```

`--no-allow-unauthenticated` keeps the URL behind Cloud Run IAM. The application key is a second check on `/models`, `/forecast`, and `/predict`. `/health` and `/version` do not require the application key so the platform probe can call `/health`. They return no forecast.

Resource choices: 1 vCPU, 4 GiB, request timeout 60 s, concurrency 1, minimum instances 0, maximum instances 2. The 60 s cloud timeout matches the in-process job limit. The F1 checkpoint is 57 KB. The memory request is for the PyTorch library, not the network. Request-based billing stays at the Cloud Run default so idle instances are not charged. Published request-based list prices, from the Cloud Run pricing page as checked on 11 October 2026, are $0.000024 per vCPU-second, $0.0000025 per GiB-second, and $0.40 per million requests, with a monthly free tier of 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million requests. Artifact Registry storage above 0.5 GiB-month is listed at $0.000136986 per GiB-hour. A multi-GB torch image left in the registry is on the order of a dollar per month. This is not a quote.

Logs go to Cloud Logging. The process redacts `WILDFIREPINN_API_KEY` if it appears in an application log line. Do not add the key to the request path.

Teammates who may invoke the service:

```text
gcloud run services add-iam-policy-binding wildfirepinn-research --region=REGION --member=user:TEAMMATE_EMAIL --role=roles/run.invoker
```

A remote call still needs an identity token and the API key:

```text
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" -H "X-API-Key: $WILDFIREPINN_API_KEY" -H "Content-Type: application/json" -d @deployment/wildfirepinn_api/examples/synthetic_circle_request.json SERVICE_URL/forecast
```

Confirm the JSON has a valid polygon, `model_id` `ray_inverse_solver`, `model_version` `experiment3-ray-v1`, units in meters and seconds, and `operational_wildfire_service` false. Repeat with a missing key and expect 401. Then POST the PINN body from `TEAM_HANDOFF.md` and expect `model_version` `experiment2a3-f1-seed0` and grid shape `[201, 201]`.

Set the client URL after that succeeds:

```powershell
$env:WILDFIREPINN_API_URL = "https://SERVICE_URL"
$env:WILDFIREPINN_API_KEY = "the secret value"
```

Cloud Run IAM identity tokens are not added by `client.forecast`. For a locked-down Cloud Run service, put an identity-token proxy in front or call `curl` as above. The Python client sends `X-API-Key` only.

## Vertex AI, if the team specifically asks for it

Do not point Vertex at this image and expect `endpoint.predict()` to work without the adapter route. `POST /predict` accepts up to four `instances`, each shaped like a `/forecast` body, and returns `{"predictions": [...]}`. Health remains `GET /health`. The container listens on `AIP_HTTP_PORT` when that variable is set.

Additional APIs and a custom container upload are a separate bill. Minimum extra steps, still not run:

```text
gcloud services enable aiplatform.googleapis.com
gcloud ai models upload --region=REGION --display-name=wildfirepinn-research-0.1.1 --container-image-uri=REGION-docker.pkg.dev/PROJECT/wildfirepinn/research-api:0.1.1 --container-health-route=/health --container-predict-route=/predict --container-ports=8080
gcloud ai endpoints create --region=REGION --display-name=wildfirepinn-research
gcloud ai endpoints deploy-model ENDPOINT_ID --region=REGION --model=MODEL_ID --display-name=ray-hybrid --machine-type=n1-standard-4 --min-replica-count=1 --max-replica-count=1
```

`n1-standard-4` is 4 vCPU and 15 GiB and does not scale to zero. That is a standing charge. Cloud Run remains the fit for an authenticated HTTP API. For Vertex, pass `--container-env-vars=WILDFIREPINN_PREDICT_AUTH=vertex_iam` only together with endpoint IAM. Image 0.1.1 starts in that mode without `WILDFIREPINN_API_KEY`. `POST /predict` then serves the Vertex envelope under platform IAM. `GET /models` and `POST /forecast` return 401 until a key of at least 16 characters is also configured. Do not set `vertex_iam` on a service that allows public invocations.

After deploy, the verification call is `rawPredict` or `predict`, then the same geometry, version, 401, and PINN checks as Cloud Run. Record latency from that response. Do not mark the service deployed until that remote call returns a polygon.
