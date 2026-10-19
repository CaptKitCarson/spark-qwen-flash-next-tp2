#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${BUILD_ROOT:-"$HERE/../.build"}
(( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) >= 90 * 1024 * 1024 )) || {
  echo 'Pause GPU workloads first: this build needs 90 GiB available.' >&2; exit 1;
}
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
export PATH="$HERE/bin:$PATH"
if [[ ! -d "$ROOT/recipe/.git" ]]; then
  git init -q "$ROOT/recipe"
  git -C "$ROOT/recipe" fetch --depth 1 https://github.com/local-inference-lab/blackwell-llm-docker.git 3d1ccd84c34ff2de6ad79004464d11474df80cb1
  git -C "$ROOT/recipe" checkout --detach FETCH_HEAD
fi
test "$(git -C "$ROOT/recipe" rev-parse HEAD)" = 3d1ccd84c34ff2de6ad79004464d11474df80cb1
if ! /usr/bin/docker buildx inspect lil-wheel-cu134-arm64 >/dev/null 2>&1; then
  /usr/bin/docker buildx create --name lil-wheel-cu134-arm64 --driver docker-container --platform linux/arm64 --driver-opt memory=80g,memory-swap=80g,cpu-period=100000,cpu-quota=1200000
fi
/usr/bin/docker buildx inspect --bootstrap lil-wheel-cu134-arm64
export BUILD_JOBS=8 MAX_JOBS=8 NVCC_THREADS=1 LIL_SPARK_CACHE="$ROOT/cache"
python3 -u "$ROOT/recipe/tools/jovian_wheel_runtime/spark_release.py" \
  --release-tag karmic-kraken-beta-08edd314f7ee880c5cc29eb4692afc2aa7a2c51736c188335b655cdbf5ed6eea \
  --work "$ROOT/work-$(date +%Y%m%d-%H%M%S)" --build-only
docker tag ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261016.2 local/lil-qwen38:arm64-20261016.2
