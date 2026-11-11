#!/usr/bin/env bash
# Runs the 4-config matrix with the 4090's Triton containers paused; always unpauses on exit.
set -u
cd "$(dirname "$0")/.."
CONTAINERS="omnivoice_triton_test fast_whisper_triton_test"
export PATH=/root/miniconda3/envs/leaptalk/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID
# Pro (Wan VAE) peaks ~9.9 GB; paused containers keep ~13.2 GB, so avoid allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
resume() { docker unpause $CONTAINERS >/dev/null 2>&1; echo "RESUMED containers $(date +%T)"; }
trap resume EXIT INT TERM
docker pause $CONTAINERS && echo "PAUSED containers $(date +%T)"
sleep 3
CONFIGS=${CONFIGS:-"lite_c0:1:off lite_c1:1:on pro_c0:0:off pro_c1:0:on"}
for cfg in $CONFIGS; do
  cfg=${cfg//:/ }
  set -- $cfg
  echo "START $1 $(date +%T)"
  # config format tag:lite:compile[:size[:repeats[:fp8[:scope]]]]
  # (size defaults to 512, repeats to $REPEATS or 2, fp8 to 0, scope to default; e.g. lite_c1_fp8:1:on:512:2:1)
  python bench/bench.py --tag "$1" --lite "$2" --compile "$3" --size "${4:-512}" --repeats "${5:-${REPEATS:-2}}" \
    --fp8 "${6:-0}" --fp8_scope "${7:-default}" > "logs/bench_$1.log" 2>&1 \
    && echo "DONE $1 $(date +%T)" || echo "FAILED $1 (see logs/bench_$1.log)"
done
echo ALL_DONE
