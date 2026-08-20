#
# Container for the DOCR-MARL thesis training/experimentation pipeline.
# The codebase is a hand-rolled NumPy/SciPy stack trained via evolution
# strategies. It runs entirely on CPU by default; marl/marl_agents.py also
# has an optional GPU-accelerated code path (auto-detected at runtime, see
# its module-level comment) that activates only if torch + a CUDA device
# are both present — see EXPERIMENT_GUIDE.md for what the experiments
# themselves do and why GPU support is opt-in at build time (below).
#
# Build (CPU-only, default — smaller image, no CUDA libraries downloaded):
#   docker build -t docr-marl .
# Build (GPU-capable — pulls torch's CUDA-bundled wheel, much larger image):
#   docker build --build-arg INSTALL_TORCH=true -t docr-marl-gpu .
#   docker run --gpus all ...   # requires nvidia-container-toolkit on the host
# Run:
#   docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
#       docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
#
# See "Running in Docker / on the cloud" in EXPERIMENT_GUIDE.md for details.

ARG PYTHON_VERSION=3.11
ARG INSTALL_TORCH=false

# ── Builder stage: compile any sdists into wheels in an isolated venv ───────
FROM python:${PYTHON_VERSION}-slim AS builder
ARG INSTALL_TORCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
# torch is listed in requirements.txt as optional (see its comment there)
# but is excluded from the default build here — its CUDA-bundled wheel is
# large, and on this workload the plain NumPy path already benchmarks
# faster on CPU anyway (see EXPERIMENT_GUIDE.md §2.2.2/§2.2.4), so most
# deployments of this image gain nothing from paying that download cost.
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v '^torch' requirements.txt > requirements-base.txt \
    && pip install --no-cache-dir -r requirements-base.txt \
    && if [ "$INSTALL_TORCH" = "true" ]; then pip install --no-cache-dir torch; fi

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

# Copies app code plus results/core_cache if present at build time (it's
# not excluded by .dockerignore — everything else under results/ and
# outputs/ is). A second copy is stashed at /opt/seed-cache so
# docker/entrypoint.sh can still seed a fresh volume even after something
# else gets bind-mounted over /app/results. `mkdir -p` on the source
# first means this is a harmless no-op when core_cache is empty/absent
# (e.g. right after a `git rm -r results/core_cache` reset) rather than
# a build failure.
COPY --chown=docr:docr . .
RUN mkdir -p /app/results/core_cache /opt/seed-cache \
    && cp -r /app/results/core_cache/. /opt/seed-cache/ \
    && chown -R docr:docr /opt/seed-cache \
    && chmod +x docker/entrypoint.sh

USER docr

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python3", "run_thesis_experiments.py", "--help"]
