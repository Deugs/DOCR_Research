#
# Container for the DOCR-MARL thesis training/experimentation pipeline.
# As of 2026-10-22, the MARL training stack (marl/marl_agents.py's
# actor/critic, with the GNN comm module and EWC estimator following the
# same pattern) is torch.nn.Module-backed and trained via real backprop +
# Adam — torch is a REQUIRED dependency, not optional (see
# marl/marl_agents.py's module-level comment for the ES-era history this
# replaced). `device="cpu"` still works fine with no GPU present; the
# build-arg below only chooses which torch WHEEL to install, not whether
# to install it at all.
#
# Build (CPU-only torch wheel, default — smaller image, no CUDA libraries):
#   docker build -t docr-marl .
# Build (CUDA-bundled torch wheel, for a GPU host — much larger image):
#   docker build --build-arg INSTALL_TORCH_CUDA=true -t docr-marl-gpu .
#   docker run --gpus all ...   # requires nvidia-container-toolkit on the host
# Run:
#   docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/outputs:/app/outputs" \
#       docr-marl python3 run_thesis_experiments.py --all --preset thesis_sweep_fast
#
# See "Running in Docker / on the cloud" in EXPERIMENT_GUIDE.md for details.

ARG PYTHON_VERSION=3.11
ARG INSTALL_TORCH_CUDA=false

# ── Builder stage: compile any sdists into wheels in an isolated venv ───────
FROM python:${PYTHON_VERSION}-slim AS builder
ARG INSTALL_TORCH_CUDA

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
# torch is REQUIRED (see requirements.txt's comment) -- the only choice
# here is which wheel: the default CPU-only build pulls the small
# CPU-only torch wheel (no CUDA runtime libraries bloating the image for
# hosts that will never have a GPU); the GPU build arg instead installs
# the default PyPI wheel, which bundles CUDA runtime libraries.
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v '^torch' requirements.txt > requirements-base.txt \
    && pip install --no-cache-dir -r requirements-base.txt \
    && if [ "$INSTALL_TORCH_CUDA" = "true" ]; then \
         pip install --no-cache-dir torch; \
       else \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
       fi

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
# outputs/ is; .venv/ and similar local virtualenvs are excluded too, see
# .dockerignore). A second copy is stashed at /opt/seed-cache so
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
