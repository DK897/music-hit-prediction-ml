import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def build_logistic(C=1.0, l1_ratio=0.5):
    return LogisticRegression(max_iter=2000, solver='saga', penalty='elasticnet',
                              l1_ratio=l1_ratio, C=C, random_state=42)

def build_lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

def build_svm():
    return SVC(kernel='rbf', probability=True, random_state=42)

def build_rf(n_estimators=300):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)

def build_gb(n_estimators=300):
    return GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

def build_nn(input_dim, hidden_sizes=(256,128), dropout=0.45, l2_reg=1e-4):
    model = Sequential([
        Dense(hidden_sizes[0], activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(hidden_sizes[1], activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
