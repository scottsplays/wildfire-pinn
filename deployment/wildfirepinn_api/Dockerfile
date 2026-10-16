FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    WILDFIREPINN_HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/wildfirepinn_api/requirements.txt /tmp/api-requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/api-requirements.txt

COPY wildfire_pinn /app/wildfire_pinn
COPY experiments/__init__.py /app/experiments/__init__.py
COPY experiments/wildfirepinn_v2_diagnostics/__init__.py /app/experiments/wildfirepinn_v2_diagnostics/__init__.py
COPY experiments/wildfirepinn_v2_diagnostics/solver.py /app/experiments/wildfirepinn_v2_diagnostics/solver.py
COPY experiments/wildfirepinn_v2_diagnostics/circle.py /app/experiments/wildfirepinn_v2_diagnostics/circle.py
COPY experiments/wildfirepinn_v2_diagnostics/ansatz.py /app/experiments/wildfirepinn_v2_diagnostics/ansatz.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/__init__.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/__init__.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/fields.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/fields.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/physics.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/physics.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/metrics.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/metrics.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/sdf.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/sdf.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/cause_isolation_2/__init__.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/cause_isolation_2/__init__.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/cause_isolation_2/models.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/cause_isolation_2/models.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/__init__.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/__init__.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/net.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/net.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/banks.py /app/experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/banks.py
COPY experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/runs/F1_seed0/levelset_pinn.pt \
    /app/experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/runs/F1_seed0/levelset_pinn.pt
COPY experiments/wildfirepinn_v3/__init__.py /app/experiments/wildfirepinn_v3/__init__.py
COPY experiments/wildfirepinn_v3/experiment3/__init__.py /app/experiments/wildfirepinn_v3/experiment3/__init__.py
COPY experiments/wildfirepinn_v3/experiment3/inverse.py /app/experiments/wildfirepinn_v3/experiment3/inverse.py
COPY experiments/wildfirepinn_v3/experiment3/models.py /app/experiments/wildfirepinn_v3/experiment3/models.py
COPY experiments/wildfirepinn_v3/experiment3/observations.py /app/experiments/wildfirepinn_v3/experiment3/observations.py
COPY experiments/wildfirepinn_v3/experiment3/torch_solver.py /app/experiments/wildfirepinn_v3/experiment3/torch_solver.py
COPY experiments/wildfirepinn_v3/experiment3/configs.json /app/experiments/wildfirepinn_v3/experiment3/configs.json
COPY deployment /app/deployment

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('AIP_HTTP_PORT') or os.environ.get('PORT') or '8080'; urllib.request.urlopen('http://127.0.0.1:%s/health' % p)"
CMD ["python", "-m", "deployment.wildfirepinn_api"]
