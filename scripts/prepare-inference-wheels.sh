#!/usr/bin/env bash
# Reproduce the wheel bytes pinned before the SDK distributions were unified.
set -euo pipefail
platform=$1
sdk=$2
output=$3
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/source" "$scratch/wheels"
export PYTHONHASHSEED=0

# The original archive/build modes and ZIP timestamps are part of the wheel hashes.
umask 0002
contracts=(apps/_shared/{inference_contract,lifecycle_contract,protocol_v2,task_contract,training_contract})
git -C "$platform" -c tar.umask=0002 archive HEAD "${contracts[@]}" |
  tar -xf - -C "$scratch/source"
# The public integration commit contains the same package trees as the original
# qualification commit c4c3913b3695ee84b4faaa9951faa3c0bd401ef3.
export SOURCE_DATE_EPOCH=1794993045
for package in "${contracts[@]}"; do
  python -m build --wheel --no-isolation --outdir "$scratch/wheels" "$scratch/source/$package"
done

packages=(packages/inference/{sdk,mask-utils})
git -C "$sdk" -c tar.umask=0022 archive HEAD "${packages[@]}" |
  tar -xf - -C "$scratch/source"
SOURCE_DATE_EPOCH=$(git -C "$sdk" show -s --format=%ct HEAD)
for package in "${packages[@]}"; do
  python -m build --wheel --no-isolation --outdir "$scratch/wheels" "$scratch/source/$package"
done

python scripts/fetch-contracts.py \
  --manifest https://github.com/YOAO-Labs/Plugin-SDK/releases/download/platform-ci-runtime-8daeb0cf02ac/manifest.json \
  --sha256 5af0e5e8c3cb68436446b2f917193a024771ec2daf94de29fc35056da75ef8a5 \
  --package yoao-backend-runtime --output "$scratch/wheels"

cp .github/actions/stage-inference-wheels/manifest.json "$scratch/wheels/"
python scripts/fetch-contracts.py --manifest "$scratch/wheels/manifest.json" \
  --sha256 "$INFERENCE_MANIFEST_SHA256" --output "$output"
cp "$scratch/wheels/manifest.json" "$output/"
