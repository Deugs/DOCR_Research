#!/bin/sh
# Seeds /app/results/core_cache from the image's baked-in copy on first run.
#
# Why this exists: results/core_cache holds the already-trained Core A/B/C
# results (~668KB, checked into git — see EXPERIMENT_GUIDE.md §2.2.1). When
# you bind-mount a host directory over /app/results for persistence (so
# training output survives past the container's life), an empty host
# directory shadows that baked-in cache and the pipeline would silently
# retrain Cores A/B/C from scratch. This copies the seed cache in exactly
# once, the first time the mounted volume is empty, then gets out of the way.
set -e

SEED_CACHE=/opt/seed-cache
TARGET_CACHE=/app/results/core_cache

if [ -d "$SEED_CACHE" ] && [ -z "$(ls -A "$TARGET_CACHE" 2>/dev/null)" ]; then
    echo "[entrypoint] $TARGET_CACHE is empty — seeding from image defaults (Core A/B/C cache)..."
    mkdir -p "$TARGET_CACHE"
    cp -r "$SEED_CACHE"/. "$TARGET_CACHE"/
fi

exec "$@"
