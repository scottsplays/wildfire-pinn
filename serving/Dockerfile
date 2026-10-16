FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    WILDFIREPINN_AUTH_MODE=iam \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt /tmp/serving-requirements.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/serving-requirements.txt

COPY wildfire_pinn /app/wildfire_pinn
COPY serving/__init__.py serving/__main__.py serving/app.py serving/auth.py serving/inference.py /app/serving/
COPY data/kincade_stack.nc data/kincade_R.nc data/kincade_perimeters.geojson /app/data/
COPY outputs/kincade_final_candidates/observation/levelset_pinn.pt \
    /app/outputs/kincade_final_candidates/observation/levelset_pinn.pt

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["python", "-m", "serving"]
