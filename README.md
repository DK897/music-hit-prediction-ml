# 🎵 Predicting Hit Songs Using Spotify Hit Predictor Dataset 🎶

### 🧠 Course: UE23CS352A—Machine Learning

### 👨‍💻 Author: Dharshan K

### 🏫 Institution: PES University  

### 📅 Year: 2025  

---

## 🌟 Project Overview

This project replicates the research paper **“Predicting Hit Songs Using Repeated Chorus”**,  
using the **Spotify Hit Predictor Dataset** (Kaggle) as the real-world equivalent.  

The aim is to determine whether the numerical **audio features** of a song can predict its **popularity (hit vs non-hit)**  
using a combination of **machine learning** and **deep learning** models.

---

## 🎯 Objectives

- Build a machine learning pipeline that can predict whether a song will be a “hit”.
- Use pre-extracted **Spotify audio features** such as:
  - `danceability`, `energy`, `valence`, `tempo`, `loudness`, `speechiness`, etc.
- Implement **all models used in the original paper**:
  1. Logistic Regression (Elastic-Net)
  2. Linear Discriminant Analysis (LDA)
  3. Support Vector Machine (RBF Kernel)
  4. Random Forest
  5. Gradient Boosting
  6. Neural Network (Feedforward MLP)
- Perform **PCA dimensionality reduction (95%)**
- Evaluate using multiple metrics: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, and `PR-AUC`.

---

## 🧰 Project Structure

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

markdown
Copy code

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
2️⃣ Add Dataset
Download the Kaggle dataset and unzip:

bash
Copy code
kaggle datasets download -d theoverman/the-spotify-hit-predictor-dataset -p data/processed
unzip data/processed/the-spotify-hit-predictor-dataset.zip -d data/processed
mv data/processed/SpotifyFeatures.csv data/processed/dataset.csv
Then clean it:

bash
Copy code
python3 src/prepare_dataset.py --infile data/processed/dataset.csv --out data/processed/dataset.csv
3️⃣ Verify Dataset
bash
Copy code
python3 src/verify_dataset.py
✅ Output Example:

pgsql
Copy code
📦 Loaded dataset: 41106 samples, 16 columns
✅ Verified dataset is numeric with '16' columns including 'target'.
4️⃣ Train Models
bash
Copy code
./run_experiments.sh
✅ Output Example:

pgsql
Copy code
[1] Verify dataset
✅ Verified dataset is numeric with '16' columns including 'target'.
[2] Train all models (Logistic, LDA, SVM, RF, GB, NN)
✅ Results saved in results/metrics_summary.csv
5️⃣ View Results
bash
Copy code
cat results/metrics_summary.csv
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.66	0.64	0.65	0.64	0.70
LDA	0.63	0.61	0.62	0.61	0.67
SVM (RBF)	0.70	0.68	0.70	0.69	0.74
Random Forest	0.72	0.71	0.72	0.71	0.77
Gradient Boosting	0.74	0.72	0.73	0.72	0.79
Neural Network	0.76	0.74	0.75	0.74	0.81

📊 Visualizations
F1-Score Comparison

ROC Curves

(Generated using 04_Compare_and_Visualize.ipynb)

🧠 Techniques Used
Step	Technique
Preprocessing	StandardScaler + PCA (95% variance)
Models	Logistic Regression (Elastic-Net), LDA, SVM (RBF), RF, GBM, NN
Evaluation	Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
Feature Reduction	PCA for noise minimization
Frameworks	Scikit-learn, TensorFlow/Keras, Pandas, Seaborn, YAML

📈 Key Findings
Neural Network achieved the highest overall performance (F1 ≈ 0.74, ROC-AUC ≈ 0.81)

Ensemble models (Random Forest, Gradient Boosting) performed better than linear models

PCA reduction improved stability without major loss of accuracy

Spotify features like energy, danceability, and valence showed strong correlation with popularity

🧩 Future Enhancements
Add XGBoost / LightGBM for improved boosting accuracy

Integrate feature importance visualization (SHAP)

Use deep audio embeddings (MFCCs) extracted from raw audio for richer feature space

Build a web dashboard for song upload and hit prediction

🏁 Conclusion
This project successfully demonstrates that audio feature-based machine learning models can predict hit songs with significant accuracy.
It replicates the methodology from “Predicting Hit Songs Using Repeated Chorus” and extends it using Spotify’s real-world feature data.