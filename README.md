# 🎵 Predicting Hit Songs Using Spotify Hit Predictor Dataset 🎶

## 🧠 Course: UE23CS352A — Machine Learning

### 👨‍💻 Author 1: B. Goutham — SRN: PES1UG23CS132  

### 👨‍💻 Author 2: Dharshan K — SRN: PES1UG23CS184  

### 🏫 Institution: PES University  

### 📅 Batch: 2023–2027  

---

## 🌟 Project Overview

This project replicates the research paper **“Predicting Hit Songs Using Repeated Chorus”**,  
using the **Spotify Hit Predictor Dataset** (Kaggle) as a real-world equivalent.  

The goal is to determine whether a song’s **numerical audio features** can predict its **popularity (hit vs. non-hit)**  
by leveraging a combination of **machine learning** and **deep learning** models.

---

## 🎯 Objectives

- Develop a **machine learning pipeline** to classify songs as *hits* or *non-hits*.
- Use pre-extracted **Spotify audio features**, including:
  - `danceability`, `energy`, `valence`, `tempo`, `loudness`, `speechiness`, etc.
- Implement all models used in the original paper:
  1. Logistic Regression (Elastic-Net)
  2. Linear Discriminant Analysis (LDA)
  3. Support Vector Machine (RBF Kernel)
  4. Random Forest
  5. Gradient Boosting
  6. Neural Network (Feedforward MLP)
- Apply **PCA (Principal Component Analysis)** to retain 95% variance for dimensionality reduction.
- Evaluate models using multiple metrics:
  - **Accuracy**, **Precision**, **Recall**, **F1 Score**, **ROC-AUC**, and **PR-AUC**.

---

## 🧰 Project Structure

```text

HitSongPrediction/
├── data/
│ └── processed/
│ └── dataset.csv # Cleaned Spotify dataset
├── experiments/
│ └── config.yaml # Configuration for training
├── notebooks/
│ ├── 01_EDA_and_DataQuality.ipynb
│ ├── 02_Preprocessing_and_PCA.ipynb
│ ├── 03_Models_and_Training.ipynb
│ ├── 04_Compare_and_Visualize.ipynb
│ └── 05_Final_Full_Pipeline.ipynb # (one-click end-to-end notebook)
├── results/
│ ├── metrics_summary.csv
│ └── model_checkpoints/
├── src/
│ ├── utils.py
│ ├── models.py
│ ├── evaluation.py
│ ├── training.py
│ ├── prepare_dataset.py
│ ├── verify_dataset.py
│ └── init.py
├── run_experiments.sh
├── requirements.txt
└── README.md
```

---

## 💾 Dataset

### 📊 Spotify Hit Predictor Dataset  

Source: [Kaggle — theoverman/the-spotify-hit-predictor-dataset](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset)

- 41,106 songs from Spotify  
- Each song includes both metadata and **numerical audio features**
- Target label: `target` →  
  - `1` = Hit Song  
  - `0` = Non-Hit Song

### ⚙️ Data Cleaning

The script `src/prepare_dataset.py` automatically:

- Drops non-numeric columns (`track`, `artist`, `uri`, `decade`)
- Ensures all remaining columns are numeric
- Renames label column to `target`
- Saves cleaned dataset → `data/processed/dataset.csv`

---

## 🚀 How to Run

### 1️⃣ Setup Environment

```bash
git clone <your_repo_link>
cd HitSongPrediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Add Dataset

```text
Download the Kaggle dataset and unzip:
```

```bash
kaggle datasets download -d theoverman/the-spotify-hit-predictor-dataset -p data/processed
unzip data/processed/the-spotify-hit-predictor-dataset.zip -d data/processed
mv data/processed/SpotifyFeatures.csv data/processed/dataset.csv
```

### Then clean it

```bash
python3 src/prepare_dataset.py --infile data/processed/dataset.csv --out data/processed/dataset.csv
```

### 3️⃣ Verify Dataset

```bash

python3 src/verify_dataset.py
```

### ✅ Output Example

```text
📦 Loaded dataset: 41106 samples, 16 columns
✅ Verified dataset is numeric with '16' columns including 'target'.
```

### 4️⃣ Train Models

```bash

./run_experiments.sh
```

```text

*** ✅ Output Example:

[1] Verify dataset
✅ Verified dataset is numeric with '16' columns including 'target'.
[2] Train all models (Logistic, LDA, SVM, RF, GB, NN)
✅ Results saved in results/metrics_summary.csv
```

### 5️⃣ View Results

```bash
cat results/metrics_summary.csv
```

```text
*** content:

| Unnamed: 0       |   accuracy |   precision |   recall |       f1 |      mcc |   roc_auc |   pr_auc |    brier |   tn |   fp |   fn |   tp |
|:-----------------|-----------:|------------:|---------:|---------:|---------:|----------:|---------:|---------:|-----:|-----:|-----:|-----:|
| Logistic         |   0.730953 |    0.696864 | 0.817439 | 0.752351 | 0.468986 |  0.803852 | 0.76739  | 0.178535 | 3312 | 1827 |  938 | 4200 |
| LDA              |   0.724433 |    0.680269 | 0.846828 | 0.754465 | 0.462967 |  0.801168 | 0.765134 | 0.181054 | 3094 | 2045 |  787 | 4351 |
| SVM_RBF          |   0.771431 |    0.724377 | 0.876216 | 0.793094 | 0.555203 |  0.845957 | 0.811943 | 0.156781 | 3426 | 1713 |  636 | 4502 |
| RandomForest     |   0.772015 |    0.745391 | 0.826197 | 0.783716 | 0.54726  |  0.843413 | 0.813255 | 0.159717 | 3689 | 1450 |  893 | 4245 |
| GradientBoosting |   0.76355  |    0.726877 | 0.844297 | 0.781199 | 0.53412  |  0.833143 | 0.796984 | 0.163713 | 3509 | 1630 |  800 | 4338 |
| NeuralNet        |   0.778437 |    0.742746 | 0.851888 | 0.793582 | 0.562991 |  0.851313 | 0.819028 | 0.153805 | 3623 | 1516 |  761 | 4377 |


```

### 📊 Visualizations

```text
F1-Score Comparison

## 📈 ROC Curves

*(Generated using `04_Compare_and_Visualize.ipynb`)*

## 🧠 Techniques Used

| Step              | Technique |
|-------------------|------------|
| **Preprocessing** | `StandardScaler` + `PCA` (95% variance retained) |
| **Models**        | Logistic Regression (Elastic-Net), LDA, SVM (RBF), Random Forest, Gradient Boosting, Neural Network |
| **Evaluation**    | Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC |
| **Feature Reduction** | PCA for noise minimization |
| **Frameworks**    | Scikit-learn, TensorFlow/Keras, Pandas, Seaborn, YAML |

📝 *This section highlights the preprocessing pipeline, model comparison strategy, and metrics used for performance visualization.*

```

### 📈 Key Findings

```text
🎯 Neural Network achieved the highest overall performance (F1 ≈ 0.79, ROC-AUC ≈ 0.85)  
🌲 Ensemble models (Random Forest, Gradient Boosting) outperformed linear models  
📊 PCA reduction improved model stability without significant accuracy loss  
🎵 Spotify features such as energy, danceability, and valence showed strong correlation with popularity
```

### 🧩 Future Enhancements

```text
⚡ Add XGBoost / LightGBM for improved boosting accuracy  
📈 Integrate SHAP for feature importance visualization  
🎧 Use deep audio embeddings (e.g., MFCCs) extracted from raw audio for a richer feature space  
🌐 Build an interactive web dashboard for song upload and hit prediction
```

### 🏁 Conclusion

```text
✅ This project demonstrates that audio feature-based machine learning models can effectively predict hit songs with strong accuracy.  
📚 It replicates and extends the methodology from *“Predicting Hit Songs Using Repeated Chorus”*, applying it to real-world Spotify feature data.
```

```text
💡 *These insights highlight the project’s success in combining music analytics with machine learning to uncover what makes a song a hit.*
```
