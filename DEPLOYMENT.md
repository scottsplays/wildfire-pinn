# Deploy WildfirePINN v1

This is a private inference API for the frozen Kincade checkpoint. It loads `outputs/kincade_final_candidates/observation/levelset_pinn.pt`, checks SHA-256 `93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996`, and runs that network on CPU. It does not train.

The model is an observation-only level-set baseline (`w_pde = 0`). Test IoU is about 0.13. It is not an operational wildfire forecast. Every prediction response repeats that warning.

The service was not deployed to Google Cloud in the session that added this file. The commands below are the procedure to follow after the project owner explicitly approves a billable deploy.

## What teammates call

| Method | Path | Auth | Body |
| --- | --- | --- | --- |
| GET | `/health` | Cloud Run IAM only | none |
| GET | `/model-info` | IAM and a bearer token | none |
| POST | `/predict` | IAM and a bearer token | `{"timestamp": "2019-10-24T09:05:00"}` |
| POST | `/predict/mask` | IAM and a bearer token | same JSON |

`/predict` returns JSON: model version, SHA-256, CRS (`EPSG:3310`), grid height and width, cell size, timestamp, τ, burned-cell count, and burned area in km². The raster is not in that JSON.

`/predict/mask` returns a compressed `.npz` file. `burned` is `uint8` with 1 where φ < 0. `x` and `y` are the 90 m cell centers. A copy of the JSON metadata is stored in the file as `metadata`.

Timestamps use the clock times stored on the Kincade overpasses, with no timezone suffix. The window is the first overpass through the last (2019-10-24 09:05 through 2019-10-30 10:33). Times outside that window are rejected. The grid is the existing Kincade grid only.

Normalization matches `infer_wildfirepinn_v1.py`: τ = (t − t0) / T, X = (x − xc) / L, Y = (y − yc) / L, and φ = Φ · L in meters.

## Local test

From the repository root, with the checkpoint and the three Kincade files already on disk (`data/kincade_stack.nc`, `data/kincade_R.nc`, `data/kincade_perimeters.geojson`):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r serving/requirements.txt
python -m serving.test_api
```

`python -m serving.test_api` checks the checkpoint hash, rejects missing and invalid tokens, rejects timestamps outside the window, and checks that `/predict` and `/predict/mask` match `predict_phi_meters` on the first overpass. It does not contact Google Cloud.

To start the server on this machine only:

```powershell
$env:WILDFIREPINN_AUTH_MODE = "local"
$env:WILDFIREPINN_LOCAL_TOKEN = "replace-with-16-or-more-random-characters"
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8080
```

The local token is an environment variable. It is not stored in the repo. Do not use `local` mode on Cloud Run. The process refuses to start on Cloud Run unless `WILDFIREPINN_AUTH_MODE=iam`.

```powershell
$headers = @{ Authorization = "Bearer $env:WILDFIREPINN_LOCAL_TOKEN" }
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/model-info -Headers $headers
Invoke-RestMethod http://127.0.0.1:8080/predict -Method Post -Headers $headers -ContentType "application/json" -Body '{"timestamp":"2019-10-24T09:05:00"}'
Invoke-WebRequest http://127.0.0.1:8080/predict/mask -Method Post -Headers $headers -ContentType "application/json" -Body '{"timestamp":"2019-10-24T09:05:00"}' -OutFile burned.npz
```

A request with no `Authorization` header, or with the wrong token, returns 401. Stop the local server with Ctrl+C.

Interactive API docs are turned off so a misconfigured public URL cannot offer a try-it page.

## Container

Build from the repository root. The image installs the CPU PyTorch wheel, copies the package, the three grid files, and the one frozen checkpoint, and runs as a non-root user. `firms_key.txt` is not copied.

```powershell
docker build -f serving/Dockerfile -t wildfirepinn-v1 .
docker run --rm -p 127.0.0.1:8080:8080 `
  -e WILDFIREPINN_AUTH_MODE=local `
  -e WILDFIREPINN_LOCAL_TOKEN=replace-with-16-or-more-random-characters `
  wildfirepinn-v1
```

The image default is `WILDFIREPINN_AUTH_MODE=iam`. Without `WILDFIREPINN_AUDIENCE` set to an `https://` URL, that default exits on startup instead of serving anonymously.

The inference function lives in `serving/inference.py`, separate from the FastAPI routes. A later Vertex AI custom-container deploy can call that function and leave Google-authenticated traffic in front of the same image. Vertex uses its own IAM and prediction path. Do that as a separate change. This prototype targets Cloud Run.

## Google Cloud setup

Do this only after an explicit decision to pay for Cloud Run, Artifact Registry, and image storage. Nothing here has been run against a project.

Use a project the team controls. These examples use names you must replace:

- `PROJECT_ID` — the Google Cloud project
- `REGION` — for example `us-west1`
- `wildfirepinn-v1` — the service name

```powershell
gcloud auth login
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create wildfirepinn --repository-format=docker --location=REGION
```

Build and push:

```powershell
gcloud auth configure-docker REGION-docker.pkg.dev
docker build -f serving/Dockerfile -t REGION-docker.pkg.dev/PROJECT_ID/wildfirepinn/wildfirepinn-v1:v1 .
docker push REGION-docker.pkg.dev/PROJECT_ID/wildfirepinn/wildfirepinn-v1:v1
```

Create a runtime service account first. It does not need permission to call other Google APIs, and it must not be granted `roles/run.invoker` on this service.

```powershell
gcloud iam service-accounts create wildfirepinn-runner --display-name "WildfirePINN v1 runtime"
```

Deploy the service private. `--no-allow-unauthenticated` is required. Do not grant `allUsers` or `allAuthenticatedUsers`. The first revision uses a placeholder audience so the process can start before the service URL exists. Tokens for that placeholder will not verify.

```powershell
gcloud run deploy wildfirepinn-v1 `
  --image REGION-docker.pkg.dev/PROJECT_ID/wildfirepinn/wildfirepinn-v1:v1 `
  --region REGION `
  --no-allow-unauthenticated `
  --service-account wildfirepinn-runner@PROJECT_ID.iam.gserviceaccount.com `
  --cpu 1 `
  --memory 4Gi `
  --timeout 300 `
  --min-instances 0 `
  --max-instances 2 `
  --set-env-vars "WILDFIREPINN_AUTH_MODE=iam,WILDFIREPINN_AUDIENCE=https://pending.invalid"
```

Read the URL, then point the audience at that exact URL. That update starts a new revision.

```powershell
gcloud run services describe wildfirepinn-v1 --region REGION --format="value(status.url)"
gcloud run services update wildfirepinn-v1 `
  --region REGION `
  --update-env-vars "WILDFIREPINN_AUDIENCE=https://THE-URL-FROM-DESCRIBE"
```

Confirm the IAM policy does not contain `allUsers`:

```powershell
gcloud run services get-iam-policy wildfirepinn-v1 --region REGION
```

## How a teammate authenticates

Cloud Run checks Google IAM before the request reaches the model. The container then verifies the same identity token. Both checks have to succeed.

Grant `roles/run.invoker` to each person or service account. Replace the member. Do this once per teammate, not for the public.

```powershell
gcloud run services add-iam-policy-binding wildfirepinn-v1 `
  --region REGION `
  --member="user:teammate@university.edu" `
  --role="roles/run.invoker"
```

That teammate signs in with `gcloud auth login`, then sends an identity token whose audience is the service URL. A token minted for a different audience is rejected.

```powershell
$url = gcloud run services describe wildfirepinn-v1 --region REGION --format="value(status.url)"
$token = gcloud auth print-identity-token --audiences=$url
curl.exe -H "Authorization: Bearer $token" "$url/model-info"
curl.exe -H "Authorization: Bearer $token" -H "Content-Type: application/json" -d "{\"timestamp\":\"2019-10-24T09:05:00\"}" "$url/predict"
curl.exe -H "Authorization: Bearer $token" -H "Content-Type: application/json" -d "{\"timestamp\":\"2019-10-24T09:05:00\"}" -o burned.npz "$url/predict/mask"
```

Put the token in the `Authorization: Bearer` header. Cloud Run also accepts `X-Serverless-Authorization`, but it strips that token's signature before the container sees it, and this service would then return 401. Use `Authorization`.

`gcloud run services proxy wildfirepinn-v1 --region REGION` is an alternative for a person who already has `roles/run.invoker`. The proxy attaches the identity token.

A caller without `roles/run.invoker` receives 403 from Cloud Run and never reaches the model. A caller who reaches the container without a verifiable token receives 401.

## Shut the service down

Scale-to-zero (`--min-instances 0`) avoids idle instance charges. Image storage in Artifact Registry still bills until the image is deleted, and each request bills CPU and memory for the time it runs. Cold starts are slow because the process loads PyTorch and the Kincade grid.

When the prototype is finished, delete the service and the image:

```powershell
gcloud run services delete wildfirepinn-v1 --region REGION
gcloud artifacts docker images delete REGION-docker.pkg.dev/PROJECT_ID/wildfirepinn/wildfirepinn-v1:v1 --delete-tags
```

Then open Cloud Billing for `PROJECT_ID` and confirm Cloud Run and Artifact Registry are no longer accruing cost. Delete the Artifact Registry repository too if nothing else uses it:

```powershell
gcloud artifacts repositories delete wildfirepinn --location REGION
```
