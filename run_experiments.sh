#!/usr/bin/env bash
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

CONFIG=experiments/config.yaml
RESULTS=results
mkdir -p ${RESULTS}/model_checkpoints

echo "[1] Verify dataset"
python3 src/verify_dataset.py

echo "[2] Train all models (Logistic, LDA, SVM, RF, GB, NN)"
python3 src/training.py --config ${CONFIG} --out ${RESULTS}

echo "[✅] Done! Metrics in results/metrics_summary.csv"
