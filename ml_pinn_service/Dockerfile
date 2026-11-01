# WildfirePINN inference service (model_type "pinn"), port 8082.
#   docker build -t pinn-service ml_pinn_service
#   docker run --rm -p 8082:8082 -e FIRMS_MAP_KEY=<your key> pinn-service
# Same image runs as a Vertex AI custom container (AIP_HTTP_PORT / AIP_*_ROUTE are honoured).
# Pass the FIRMS key at run time (Secret Manager on GCP); never bake it into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIP_HTTP_PORT=8082 \
    INGESTION=native \
    STATIC_LAYERS=/app/pinn_service/static/bay_area_static.npz

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY serve.py /app/serve.py
COPY pinn_service /app/pinn_service
# synthetic replay files for `-e INGESTION=npz` smoke tests without network or keys
RUN python -m pinn_service.make_test_features && rm -f /app/firms_key.txt /app/pinn_service/firms_key.txt

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"AIP_HTTP_PORT\",\"8082\")}/health')"
CMD ["sh", "-c", "python serve.py --port ${AIP_HTTP_PORT:-8082}"]
