# Local dashboard

Streamlit map for the research API at `http://127.0.0.1:8080`. Forecast polygons are the `/forecast` response. The page does not draw a perimeter until that call succeeds.

The ray model is a numerical hybrid. `direct_pinn_experimental` is the Experiment 2A.3 PINN and only accepts its training circle.

## PowerShell

Use the same API key in both terminals. It must be at least 16 characters. Do not commit it.

Terminal 1, from the repository root, if the container is not already running:

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
docker run --rm -p 127.0.0.1:8080:8080 -e WILDFIREPINN_API_KEY=$env:WILDFIREPINN_API_KEY wildfirepinn-research-api:0.1.1
```

If that image has not been built yet, the build command is in `deployment/wildfirepinn_api/README_DEPLOYMENT.md`. A local process on port 8080 is the same API:

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
python -m uvicorn deployment.wildfirepinn_api.service.app:app --host 127.0.0.1 --port 8080
```

Terminal 2, from the repository root:

```powershell
$env:WILDFIREPINN_API_KEY = "replace-with-a-long-random-string"
pip install -r deployment/wildfirepinn_api/dashboard/requirements.txt
streamlit run deployment/wildfirepinn_api/dashboard/app.py
```

Open the URL Streamlit prints. The synthetic circle is `deployment/wildfirepinn_api/examples/synthetic_circle_request.json`. Choose a model and a horizon, then click **Request forecast**.
