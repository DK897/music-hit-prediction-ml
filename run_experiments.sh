#!/usr/bin/env bash
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
CONFIG=experiments/config.yaml
RESULTS=results
mkdir -p ${RESULTS}/model_checkpoints
echo "[1] Feature extraction (if chorus_wavs available)"
python3 src/feature_extraction.py --wavs_dir data/chorus_wavs --out_csv data/features/chorus_features.csv || true
echo "[2] Prepare dataset"
python3 src/prepare_dataset.py --features data/features/chorus_features.csv --labels data/labels.csv --out data/processed/dataset.csv || true
echo "[3] Run training (classical + NN)"
python3 src/training.py --config ${CONFIG} --out ${RESULTS}
echo "[Done] Results in ${RESULTS}"
