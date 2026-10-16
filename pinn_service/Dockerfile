# WildfirePINN inference service. Build from the repository root:
#   docker build -f pinn_service/Dockerfile -t wildfirepinn-inference:0.1.0 .
#   docker run --rm -p 8080:8080 wildfirepinn-inference:0.1.0
# Same image runs as a Vertex AI custom container (AIP_HTTP_PORT / AIP_*_ROUTE are honoured).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AIP_HTTP_PORT=8080 \
    FEATURE_SOURCE=local \
    FEATURE_DIR=/app/pinn_service/test_features

WORKDIR /app
COPY pinn_service/requirements.txt /app/pinn_service/requirements.txt
RUN pip install --no-cache-dir -r /app/pinn_service/requirements.txt

COPY pinn_service /app/pinn_service
# Synthetic test files so `docker run` answers out of the box; production sets
# FEATURE_SOURCE=gcs and reads gs://fightfirewai-features/daily/YYYY-MM-DD.npz.
RUN python -m pinn_service.make_test_features

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"AIP_HTTP_PORT\",\"8080\")}/health')"
CMD ["sh", "-c", "python -m pinn_service.serve --port ${AIP_HTTP_PORT:-8080}"]
