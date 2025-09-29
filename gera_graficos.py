import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(10, 6))
class_counts = df['Class'].value_counts()
bars = ax.bar(['Legítimas', 'Fraudulentas'], class_counts.values, 
              color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(df)*100:.2f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Número de Transações', fontsize=13, fontweight='bold')
ax.set_title('Distribuição de Classes no Dataset', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, max(class_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig1_distribuicao_classes.png', 
            dpi=300, bbox_inches='tight')
plt.close()

df_metrics = pd.read_csv(r'C:\Users\dados1\Documents\tcc\tcc\resultados_metricas.csv')
df_metrics['F1-Score'] = df_metrics['F1-Score'].astype(float)

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
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig2_comparacao_metricas.png', 
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
pivot_f1 = df_metrics.pivot(index='Balanceamento', columns='Modelo', values='F1-Score')

sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', 
            linewidths=2, linecolor='black', cbar_kws={'label': 'F1-Score'},
            vmin=0, vmax=1, ax=ax, annot_kws={'fontsize': 13, 'fontweight': 'bold'})

ax.set_title('F1-Score: Comparação entre Modelos e Técnicas de Balanceamento', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Modelo', fontsize=13, fontweight='bold')
ax.set_ylabel('Técnica de Balanceamento', fontsize=13, fontweight='bold')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig3_heatmap_f1score.png', 
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))

for balance_name in results:
    for model_name in results[balance_name]:
        if balance_name == best_config['balance'] and model_name == best_config['model']:
            y_pred_proba = results[balance_name][model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = results[balance_name][model_name]['auc_roc']
            
            ax.plot(fpr, tpr, linewidth=3, 
                   label=f'{model_name} + {balance_name} (AUC = {auc:.4f})',
                   color='#e74c3c')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Classificador Aleatório', alpha=0.5)
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=13, fontweight='bold')
ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=13, fontweight='bold')
ax.set_title('Curva ROC - Melhor Modelo', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig4_curva_roc_melhor.png', 
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
colors_roc = {'Logistic Regression': '#3498db', 'Random Forest': '#e67e22', 'XGBoost': '#9b59b6'}

for model_name in models.keys():
    balance_name = best_config['balance']
    y_pred_proba = results[balance_name][model_name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = results[balance_name][model_name]['auc_roc']
    
    ax.plot(fpr, tpr, linewidth=2.5, 
           label=f'{model_name} (AUC = {auc:.4f})',
           color=colors_roc[model_name])

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Classificador Aleatório', alpha=0.5)
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=13, fontweight='bold')
ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=13, fontweight='bold')
ax.set_title(f'Comparação de Curvas ROC - {best_config["balance"]}', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig5_curvas_roc_comparacao.png', 
            dpi=300, bbox_inches='tight')
plt.close()

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
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig6_matriz_confusao_melhor.png', 
            dpi=300, bbox_inches='tight')
plt.close()

df_cv = pd.read_csv(r'C:\Users\dados1\Documents\tcc\tcc\resultados_validacao_cruzada.csv')
df_cv['F1-Score Médio'] = df_cv['F1-Score Médio'].astype(float)
df_cv['Desvio Padrão'] = df_cv['Desvio Padrão'].astype(float)

fig, ax = plt.subplots(figsize=(14, 8))
pivot_cv = df_cv.pivot(index='Balanceamento', columns='Modelo', values='F1-Score Médio')
pivot_std = df_cv.pivot(index='Balanceamento', columns='Modelo', values='Desvio Padrão')

x = np.arange(len(pivot_cv.index))
width = 0.25

for i, model in enumerate(pivot_cv.columns):
    ax.bar(x + i*width, pivot_cv[model], width, 
           yerr=pivot_std[model], capsize=5,
           label=model, color=colors[i], 
           edgecolor='black', linewidth=0.8)

ax.set_xlabel('Técnica de Balanceamento', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score Médio (5-Fold CV)', fontsize=13, fontweight='bold')
ax.set_title('Validação Cruzada: F1-Score por Modelo e Balanceamento', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot_cv.index, rotation=15, ha='right')
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig7_validacao_cruzada.png', 
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
pivot_recall = df_metrics.pivot(index='Balanceamento', columns='Modelo', values='Recall')

x = np.arange(len(pivot_recall.index))
width = 0.25

for i, model in enumerate(pivot_recall.columns):
    ax.bar(x + i*width, pivot_recall[model], width, label=model, 
           color=colors[i], edgecolor='black', linewidth=0.8)

ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
           label='Threshold Aceitável (80%)', alpha=0.7)
ax.set_xlabel('Técnica de Balanceamento', fontsize=13, fontweight='bold')
ax.set_ylabel('Recall (Sensibilidade)', fontsize=13, fontweight='bold')
ax.set_title('Impacto das Técnicas de Balanceamento no Recall', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(pivot_recall.index, rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig8_impacto_balanceamento_recall.png', 
            dpi=300, bbox_inches='tight')
plt.close()