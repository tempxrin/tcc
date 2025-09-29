"""
TCC: APLICAÇÃO DE MACHINE LEARNING NA IDENTIFICAÇÃO DE FRAUDES FINANCEIRAS
Autor: João Daniel Temporin
Orientador: Prof. Maura Velho

Script Master - Execução Completa da Análise
"""

import os
import time

print("="*80)
print("INICIANDO PROCESSAMENTO COMPLETO DO TCC")
print("="*80)
print()

# ============================================================================
# ETAPA 1: Download dos Dados
# ============================================================================
print("[ETAPA 1/4] Baixando dataset do Kaggle...")
start_time = time.time()

import kagglehub
import pandas as pd
import shutil

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# O arquivo original do Kaggle se chama "creditcard.csv"
arquivo_original = os.path.join(path, "creditcard.csv")
arquivo_destino = r'C:\Users\dados1\Documents\tcc\tcc\dados_cartao_credito.csv'

# Copiar e renomear
shutil.copy2(arquivo_original, arquivo_destino)

df = pd.read_csv(arquivo_destino)

print(f"✓ Dataset carregado: {len(df):,} transações")
print(f"✓ Tempo: {time.time() - start_time:.2f}s\n")

# ============================================================================
# ETAPA 2: Análise Exploratória
# ============================================================================
print("[ETAPA 2/4] Realizando análise exploratória...")
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Análise de desbalanceamento
class_distribution = df['Class'].value_counts()
fraud_ratio = class_distribution[1] / len(df) * 100

# Gráfico 1: Distribuição de Classes
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Legítimas', 'Fraudulentas'], class_distribution.values, 
              color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df)*100:.2f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Número de Transações', fontsize=13, fontweight='bold')
ax.set_title('Distribuição de Classes no Dataset', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, max(class_distribution.values) * 1.15)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig1_distribuicao_classes.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Análise de importância de variáveis
X = df.drop('Class', axis=1)
y = df['Class']
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=False)

# Gráfico 2: Importância das variáveis
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(15)
colors_bar = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(top_features))]
bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors_bar, 
               edgecolor='black', linewidth=1)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Variáveis Mais Importantes', fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig2_importancia_variaveis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

feature_importance.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\importancia_variaveis.csv', 
                          index=False, encoding='utf-8-sig')

print(f"✓ Análise exploratória concluída")
print(f"✓ Tempo: {time.time() - start_time:.2f}s\n")

# ============================================================================
# ETAPA 3: Modelagem e Treinamento
# ============================================================================
print("[ETAPA 3/4] Treinando modelos de Machine Learning...")
start_time = time.time()

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

# Pré-processamento
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Técnicas de balanceamento
X_train_orig, y_train_orig = X_train.copy(), y_train.copy()

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

smotetomek = SMOTETomek(random_state=42)
X_train_hybrid, y_train_hybrid = smotetomek.fit_resample(X_train, y_train)

# Modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
}

balancing_techniques = {
    'Original': (X_train_orig, y_train_orig),
    'SMOTE': (X_train_smote, y_train_smote),
    'Random Under Sampling': (X_train_rus, y_train_rus),
    'SMOTE + Tomek': (X_train_hybrid, y_train_hybrid)
}

results = {}

# Treinamento
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

# Validação cruzada
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

# Identificar melhor modelo
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

print(f"✓ Treinamento concluído")
print(f"✓ Melhor modelo: {best_config['model']} + {best_config['balance']}")
print(f"✓ F1-Score: {best_config['metrics']['f1_score']:.4f}")
print(f"✓ Tempo: {time.time() - start_time:.2f}s\n")

# ============================================================================
# ETAPA 4: Geração de Resultados e Visualizações
# ============================================================================
print("[ETAPA 4/4] Gerando relatórios e visualizações...")
start_time = time.time()

# Exportar métricas
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

# Exportar validação cruzada
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

# Gráfico 3: Comparação de métricas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
colors = ['#3498db', '#e67e22', '#9b59b6']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    df_metrics[metric] = df_metrics[metric].astype(float)
    
    pivot = df_metrics.pivot(index='Balanceamento', columns='Modelo', values=metric)
    x = np.arange(len(pivot.index))
    width = 0.25
    
    for i, model in enumerate(pivot.columns):
        ax.bar(x + i*width, pivot[model], width, label=model, color=colors[i], 
               edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Técnica de Balanceamento', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} por Modelo e Balanceamento', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig3_comparacao_metricas.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 4: Heatmap F1-Score
fig, ax = plt.subplots(figsize=(12, 8))
pivot_f1 = df_metrics.pivot(index='Balanceamento', columns='Modelo', values='F1-Score')
pivot_f1 = pivot_f1.astype(float)

sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', 
            linewidths=2, linecolor='black', cbar_kws={'label': 'F1-Score'},
            vmin=0, vmax=1, ax=ax, annot_kws={'fontsize': 13, 'fontweight': 'bold'})

ax.set_title('F1-Score: Comparação entre Modelos e Técnicas', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Modelo', fontsize=13, fontweight='bold')
ax.set_ylabel('Técnica de Balanceamento', fontsize=13, fontweight='bold')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig4_heatmap_f1score.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 5: Matriz de Confusão
cm = best_config['metrics']['confusion_matrix']
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            linewidths=3, linecolor='black', cbar_kws={'label': 'Quantidade'},
            ax=ax, annot_kws={'fontsize': 16, 'fontweight': 'bold'})

ax.set_xlabel('Predição', fontsize=13, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=13, fontweight='bold')
ax.set_title(f'Matriz de Confusão - {best_config["model"]} + {best_config["balance"]}', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticklabels(['Legítima', 'Fraude'], fontsize=12)
ax.set_yticklabels(['Legítima', 'Fraude'], fontsize=12, rotation=0)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig5_matriz_confusao.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 6: Curva ROC
fig, ax = plt.subplots(figsize=(10, 8))

y_pred_proba = best_config['metrics']['y_pred_proba']
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = best_config['metrics']['auc_roc']

ax.plot(fpr, tpr, linewidth=3, 
       label=f'{best_config["model"]} (AUC = {auc:.4f})',
       color='#e74c3c')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Classificador Aleatório', alpha=0.5)
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=13, fontweight='bold')
ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=13, fontweight='bold')
ax.set_title('Curva ROC - Melhor Modelo', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig6_curva_roc.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# Relatório final
with open(r'C:\Users\dados1\Documents\tcc\tcc\relatorio_final.txt', 'w', encoding='utf-8') as f:
    f.write("RELATÓRIO FINAL - TCC\n")
    f.write("="*70 + "\n\n")
    f.write(f"Autor: João Daniel Temporin\n")
    f.write(f"Orientador: Prof. Maura Velho\n\n")
    
    f.write("DATASET\n")
    f.write("-"*70 + "\n")
    f.write(f"Total de transações: {len(df):,}\n")
    f.write(f"Transações legítimas: {class_distribution[0]:,} ({(class_distribution[0]/len(df))*100:.2f}%)\n")
    f.write(f"Transações fraudulentas: {class_distribution[1]:,} ({(class_distribution[1]/len(df))*100:.2f}%)\n")
    f.write(f"Razão de desbalanceamento: 1:{int(class_distribution[0]/class_distribution[1])}\n\n")
    
    f.write("MELHOR MODELO\n")
    f.write("-"*70 + "\n")
    f.write(f"Algoritmo: {best_config['model']}\n")
    f.write(f"Técnica de Balanceamento: {best_config['balance']}\n\n")
    
    f.write("MÉTRICAS DE DESEMPENHO\n")
    f.write("-"*70 + "\n")
    f.write(f"Precision: {best_config['metrics']['precision']:.4f} ({best_config['metrics']['precision']*100:.2f}%)\n")
    f.write(f"Recall: {best_config['metrics']['recall']:.4f} ({best_config['metrics']['recall']*100:.2f}%)\n")
    f.write(f"F1-Score: {best_config['metrics']['f1_score']:.4f} ({best_config['metrics']['f1_score']*100:.2f}%)\n")
    f.write(f"AUC-ROC: {best_config['metrics']['auc_roc']:.4f} ({best_config['metrics']['auc_roc']*100:.2f}%)\n\n")
    
    f.write("MATRIZ DE CONFUSÃO\n")
    f.write("-"*70 + "\n")
    tn, fp, fn, tp = cm.ravel()
    f.write(f"Verdadeiros Negativos: {tn:,}\n")
    f.write(f"Falsos Positivos: {fp:,}\n")
    f.write(f"Falsos Negativos: {fn:,}\n")
    f.write(f"Verdadeiros Positivos: {tp:,}\n\n")
    
    f.write("CONCLUSÃO\n")
    f.write("-"*70 + "\n")
    f.write(f"O modelo {best_config['model']}, combinado com a técnica de balanceamento\n")
    f.write(f"{best_config['balance']}, apresentou o melhor desempenho na detecção de\n")
    f.write(f"fraudes financeiras, atingindo {best_config['metrics']['f1_score']*100:.2f}% de F1-Score e\n")
    f.write(f"{best_config['metrics']['auc_roc']*100:.2f}% de AUC-ROC. Estes resultados demonstram a eficácia\n")
    f.write(f"da aplicação de técnicas de Machine Learning aliadas a estratégias adequadas\n")
    f.write(f"de tratamento de dados desbalanceados para a identificação de transações\n")
    f.write(f"fraudulentas em bases de dados financeiros.\n")

print(f"✓ Relatórios e visualizações gerados")
print(f"✓ Tempo: {time.time() - start_time:.2f}s\n")

print("="*80)
print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print("="*80)
print("\nArquivos gerados:")
print("  - dados_cartao_credito.csv (dataset)")
print("  - resultados_metricas.csv")
print("  - resultados_validacao_cruzada.csv")
print("  - importancia_variaveis.csv")
print("  - relatorio_final.txt")
print("  - fig1_distribuicao_classes.png")
print("  - fig2_importancia_variaveis.png")
print("  - fig3_comparacao_metricas.png")
print("  - fig4_heatmap_f1score.png")
print("  - fig5_matriz_confusao.png")
print("  - fig6_curva_roc.png")
print("\nResultados:")
print(f"  Melhor Modelo: {best_config['model']}")
print(f"  Balanceamento: {best_config['balance']}")
print(f"  F1-Score: {best_config['metrics']['f1_score']:.4f}")
print(f"  AUC-ROC: {best_config['metrics']['auc_roc']:.4f}")