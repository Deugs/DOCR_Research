#
# Container for the DOCR-MARL thesis training/experimentation pipeline.
# CPU-only: the codebase is a hand-rolled NumPy/SciPy stack trained via
# evolution strategies (no PyTorch/TensorFlow, no GPU needed) — see
# EXPERIMENT_GUIDE.md for what the experiments themselves do.
#
# Build:  docker build -t docr-marl .
# Run:    docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
#             docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
#
# See "Running in Docker / on the cloud" in EXPERIMENT_GUIDE.md for details.

ARG PYTHON_VERSION=3.11

# ── Builder stage: compile any sdists into wheels in an isolated venv ───────
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage: slim image, no compilers ─────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# matplotlib (Agg backend, no display server needed) still wants a font
# available for figure text; fonts-dejavu-core is tiny and covers it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 docr

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Copies app code plus the small pre-trained results/core_cache baked in
# by .dockerignore (everything else under results/ and outputs/ is
# excluded — see that file). A second copy of core_cache is stashed at
# /opt/seed-cache so docker/entrypoint.sh can still seed a fresh volume
# even after something else gets bind-mounted over /app/results.
COPY --chown=docr:docr . .
RUN mkdir -p /opt/seed-cache \
    && cp -r /app/results/core_cache/. /opt/seed-cache/ \
    && chown -R docr:docr /opt/seed-cache \
    && chmod +x docker/entrypoint.sh

USER docr

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python3", "run_thesis_experiments.py", "--help"]
