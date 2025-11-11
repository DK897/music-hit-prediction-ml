#!/usr/bin/env python3
import os, argparse, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import read_config, seed_everything
from evaluation import compute_metrics
from models import (build_logistic, build_lda, build_svm, build_rf, build_gb, build_nn)

def main(config, out):
    cfg = read_config(config)
    seed_everything(cfg['dataset']['seed'])
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(cfg['dataset']['path'])
    # normalize lower-case column names
    df.columns = [c.strip() for c in df.columns]
    label_col = cfg['dataset']['label_column']
    if label_col not in df.columns:
        # try alternative names
        if 'target' in df.columns: label_col = 'target'
        elif 'label' in df.columns: label_col = 'label'
        else:
            raise KeyError(f"Label column {cfg['dataset']['label_column']} not found in dataset.")

    X = df.drop(columns=[label_col]).values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg['dataset']['test_fraction'], stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    if cfg['training']['use_pca']:
        pca = PCA(n_components=cfg['training']['pca_variance'], svd_solver='full').fit(X_train_s)
        X_train_p = pca.transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
    else:
        pca = None
        X_train_p, X_test_p = X_train_s, X_test_s

    results = {}

    # Logistic Regression
    log = build_logistic(C=cfg['models']['logistic']['C'], l1_ratio=cfg['models']['logistic']['l1_ratio'])
    log.fit(X_train_p, y_train)
    y_pred = log.predict(X_test_p)
    y_prob = log.predict_proba(X_test_p)[:,1]
    results['Logistic'] = compute_metrics(y_test, y_pred, y_prob)
    joblib.dump({'model':log,'scaler':scaler,'pca':pca}, os.path.join(out,'logistic.joblib'))

    # LDA
    lda = build_lda()
    lda.fit(X_train_p, y_train)
    y_pred = lda.predict(X_test_p)
    y_prob = lda.predict_proba(X_test_p)[:,1]
    results['LDA'] = compute_metrics(y_test, y_pred, y_prob)
    joblib.dump({'model':lda,'scaler':scaler,'pca':pca}, os.path.join(out,'lda.joblib'))

    # SVM (RBF)
    svm = build_svm()
    svm.fit(X_train_p, y_train)
    y_pred = svm.predict(X_test_p)
    try:
        y_prob = svm.predict_proba(X_test_p)[:,1]
    except Exception:
        # fallback to decision function scaled
        dfcn = svm.decision_function(X_test_p)
        y_prob = (dfcn - dfcn.min()) / (dfcn.max() - dfcn.min())
    results['SVM_RBF'] = compute_metrics(y_test, y_pred, y_prob)
    joblib.dump({'model':svm,'scaler':scaler,'pca':pca}, os.path.join(out,'svm.joblib'))

    # Random Forest
    rf = build_rf(n_estimators=cfg['models']['rf']['n_estimators'])
    rf.fit(X_train_p, y_train)
    y_pred = rf.predict(X_test_p)
    y_prob = rf.predict_proba(X_test_p)[:,1]
    results['RandomForest'] = compute_metrics(y_test, y_pred, y_prob)
    joblib.dump({'model':rf,'scaler':scaler,'pca':pca}, os.path.join(out,'rf.joblib'))

    # Gradient Boosting
    gb = build_gb(n_estimators=cfg['models']['gb']['n_estimators'])
    gb.fit(X_train_p, y_train)
    y_pred = gb.predict(X_test_p)
    y_prob = gb.predict_proba(X_test_p)[:,1]
    results['GradientBoosting'] = compute_metrics(y_test, y_pred, y_prob)
    joblib.dump({'model':gb,'scaler':scaler,'pca':pca}, os.path.join(out,'gb.joblib'))

    # Neural Network (Keras)
    input_dim = X_train_p.shape[1]
    nn = build_nn(input_dim, hidden_sizes=tuple(cfg['models']['nn']['hidden_sizes']),
                  dropout=cfg['models']['nn']['dropout'], l2_reg=cfg['models']['nn']['l2_reg'])
    nn.fit(X_train_p, y_train, validation_split=0.2,
           epochs=cfg['models']['nn']['epochs'], batch_size=cfg['models']['nn']['batch_size'], verbose=1)
    y_prob = nn.predict(X_test_p).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    results['NeuralNet'] = compute_metrics(y_test, y_pred, y_prob)
    nn.save(os.path.join(out, 'nn_model.h5'))

    # Save results
    pd.DataFrame(results).T.to_csv(os.path.join(out, 'metrics_summary.csv'))
    print(json.dumps(results, indent=2))
    print("✅ Results saved to", out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/config.yaml')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    main(args.config, args.out)
