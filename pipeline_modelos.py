import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\dados1\Documents\tcc\tcc\dados_cartao_credito.csv')

class_distribution = df['Class'].value_counts()
fraud_ratio = class_distribution[1] / len(df) * 100

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_orig = X_train.copy()
y_train_orig = y_train.copy()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

smotetomek = SMOTETomek(random_state=42)
X_train_hybrid, y_train_hybrid = smotetomek.fit_resample(X_train, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

balancing_techniques = {
    'Original': (X_train_orig, y_train_orig),
    'SMOTE': (X_train_smote, y_train_smote),
    'Random Under Sampling': (X_train_rus, y_train_rus),
    'SMOTE + Tomek': (X_train_hybrid, y_train_hybrid)
}

results = {}

for balance_name, (X_bal, y_bal) in balancing_techniques.items():
    results[balance_name] = {}
    
    for model_name, model in models.items():
        model.fit(X_bal, y_bal)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[balance_name][model_name] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for balance_name, (X_bal, y_bal) in balancing_techniques.items():
    cv_results[balance_name] = {}
    
    for model_name, model in models.items():
        scores = cross_val_score(model, X_bal, y_bal, cv=skf, 
                                scoring='f1', n_jobs=-1)
        cv_results[balance_name][model_name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }

best_f1 = 0
best_config = {}

for balance_name in results:
    for model_name in results[balance_name]:
        f1 = results[balance_name][model_name]['f1_score']
        if f1 > best_f1:
            best_f1 = f1
            best_config = {
                'balance': balance_name,
                'model': model_name,
                'metrics': results[balance_name][model_name]
            }

metrics_data = []
for balance in results:
    for model in results[balance]:
        metrics_data.append({
            'Balanceamento': balance,
            'Modelo': model,
            'Precision': f"{results[balance][model]['precision']:.4f}",
            'Recall': f"{results[balance][model]['recall']:.4f}",
            'F1-Score': f"{results[balance][model]['f1_score']:.4f}",
            'AUC-ROC': f"{results[balance][model]['auc_roc']:.4f}"
        })

df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\resultados_metricas.csv', 
                  index=False, encoding='utf-8-sig')

cv_data = []
for balance in cv_results:
    for model in cv_results[balance]:
        cv_data.append({
            'Balanceamento': balance,
            'Modelo': model,
            'F1-Score Médio': f"{cv_results[balance][model]['mean']:.4f}",
            'Desvio Padrão': f"{cv_results[balance][model]['std']:.4f}"
        })

df_cv = pd.DataFrame(cv_data)
df_cv.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\resultados_validacao_cruzada.csv', 
             index=False, encoding='utf-8-sig')