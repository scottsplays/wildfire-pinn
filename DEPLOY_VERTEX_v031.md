# Deploy WildfirePINN v0.3.1 to Vertex AI (personal Google Cloud project)

Prerequisites (once): billing enabled on the project; Google Cloud CLI installed (`gcloud --version`).
Docker is NOT needed (Cloud Build builds the image). Push v0.3.1 to the team branch first: the build uses
`CMPE295A-Project\ml_pinn_service`.

```powershell
$PROJECT = "my-project-id"          # the project ID (not the display name), from the console home page
$REGION  = "us-central1"
$IMAGE   = "$REGION-docker.pkg.dev/$PROJECT/wildfire/pinn-service:0.3.1"

gcloud auth login
gcloud config set project $PROJECT
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create wildfire --repository-format=docker --location=$REGION

# 1. build the container in the cloud (~5-10 min)
cd C:\Users\14158\Downloads\CMPE295A-Project\ml_pinn_service
gcloud builds submit --tag $IMAGE .

# 2. register it as a Vertex AI model (FIRMS key as an environment variable; never in the repo)
$KEY = (Get-Content C:\Users\14158\Downloads\WildfirePINN\firms_key.txt | Where-Object { $_ -and -not $_.StartsWith("#") } | Select-Object -First 1).Trim()
gcloud ai models upload --region=$REGION --display-name=pinn-fire-spread --container-image-uri=$IMAGE `
  --container-ports=8082 --container-health-route=/health --container-predict-route=/predict `
  --container-env-vars="FIRMS_MAP_KEY=$KEY,INGESTION=native"
gcloud ai models list --region=$REGION        # copy MODEL_ID

# 3. endpoint + deploy (~15-20 min)
gcloud ai endpoints create --region=$REGION --display-name=pinn-fire-spread-v1
gcloud ai endpoints list --region=$REGION     # copy ENDPOINT_ID
gcloud ai endpoints deploy-model ENDPOINT_ID --region=$REGION --model=MODEL_ID `
  --display-name=pinn-v031 --machine-type=n1-standard-2 --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100
```

Test (save as request.json):
`{"instances": [{"bbox": [-122.9, 36.9, -121.5, 38.3], "time_horizon": 6, "model_type": "pinn"}]}`
then `gcloud ai endpoints predict ENDPOINT_ID --region=$REGION --json-request=request.json`

Notes
- Cost: n1-standard-2 ~ $2-3/day (~$70-90/month) while deployed, even idle. Set a budget alert.
  Pause: `gcloud ai endpoints undeploy-model ENDPOINT_ID --region=$REGION --deployed-model-id=...`
- Team access: grant their account / service account "Vertex AI User" on this project (IAM), then give them
  VERTEX_AI_PROJECT_ID, VERTEX_AI_LOCATION=us-central1, VERTEX_PINN_ENDPOINT_ID.
- Fuel coverage is the Bay Area (+~25 km); fires elsewhere get a "no fuel data" warning.

Next after deployment: combined PINN/U-Net model with teammate (ask: what the U-Net predicts, grid and
horizons, training years (must exclude 2026 fires for a fair test), inputs). RL in November.
