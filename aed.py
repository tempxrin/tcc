import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\dados1\Documents\tcc\tcc\creditcard.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].boxplot([df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']], 
                labels=['Legítimas', 'Fraudulentas'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))
axes[0].set_ylabel('Valor da Transação (€)', fontsize=12, fontweight='bold')
axes[0].set_title('Distribuição do Valor das Transações', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

df[df['Class']==0]['Amount'].hist(bins=50, alpha=0.7, color='green', 
                                   label='Legítimas', ax=axes[1], edgecolor='black')
df[df['Class']==1]['Amount'].hist(bins=50, alpha=0.7, color='red', 
                                   label='Fraudulentas', ax=axes[1], edgecolor='black')
axes[1].set_xlabel('Valor da Transação (€)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequência', fontsize=12, fontweight='bold')
axes[1].set_title('Histograma: Valor das Transações', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_xlim(0, 500)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig9_distribuicao_amount.png', 
            dpi=300, bbox_inches='tight')
plt.close()

df['Time_hours'] = df['Time'] / 3600

fig, ax = plt.subplots(figsize=(14, 6))
fraud_time = df[df['Class']==1]['Time_hours']
legit_time = df[df['Class']==0]['Time_hours']

ax.hist(legit_time, bins=48, alpha=0.6, color='green', 
        label='Legítimas', edgecolor='black', linewidth=0.5)
ax.hist(fraud_time, bins=48, alpha=0.8, color='red', 
        label='Fraudulentas', edgecolor='black', linewidth=0.8)

ax.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
ax.set_ylabel('Número de Transações', fontsize=12, fontweight='bold')
ax.set_title('Distribuição Temporal das Transações', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig10_distribuicao_temporal.png', 
            dpi=300, bbox_inches='tight')
plt.close()

fraud_data = df[df['Class']==1].drop(['Time', 'Class'], axis=1)
corr_matrix = fraud_data.corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={'label': 'Correlação'},
            ax=ax, square=True)
ax.set_title('Matriz de Correlação - Transações Fraudulentas', 
             fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig11_correlacao_fraudes.png', 
            dpi=300, bbox_inches='tight')
plt.close()

stats_comparison = pd.DataFrame({
    'Variável': ['Amount', 'Time'],
    'Média Legítimas': [
        df[df['Class']==0]['Amount'].mean(),
        df[df['Class']==0]['Time'].mean()
    ],
    'Média Fraudulentas': [
        df[df['Class']==1]['Amount'].mean(),
        df[df['Class']==1]['Time'].mean()
    ],
    'Mediana Legítimas': [
        df[df['Class']==0]['Amount'].median(),
        df[df['Class']==0]['Time'].median()
    ],
    'Mediana Fraudulentas': [
        df[df['Class']==1]['Amount'].median(),
        df[df['Class']==1]['Time'].median()
    ],
    'Desvio Padrão Legítimas': [
        df[df['Class']==0]['Amount'].std(),
        df[df['Class']==0]['Time'].std()
    ],
    'Desvio Padrão Fraudulentas': [
        df[df['Class']==1]['Amount'].std(),
        df[df['Class']==1]['Time'].std()
    ]
})

stats_comparison.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\estatisticas_comparativas.csv', 
                        index=False, encoding='utf-8-sig')

from sklearn.ensemble import RandomForestClassifier

X = df.drop('Class', axis=1)
y = df['Class']

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(15)
colors_bar = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(top_features))]

bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors_bar, 
               edgecolor='black', linewidth=1)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importância', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Variáveis Mais Importantes para Detecção de Fraudes', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig12_importancia_variaveis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

feature_importance.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\importancia_variaveis.csv', 
                          index=False, encoding='utf-8-sig')

top_v_features = [f for f in feature_importance.head(4)['Feature'] if f.startswith('V')]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_v_features):
    ax = axes[idx]
    
    legit = df[df['Class']==0][feature]
    fraud = df[df['Class']==1][feature]
    
    ax.hist(legit, bins=50, alpha=0.6, color='green', label='Legítimas', 
            edgecolor='black', linewidth=0.5, density=True)
    ax.hist(fraud, bins=50, alpha=0.7, color='red', label='Fraudulentas', 
            edgecolor='black', linewidth=0.8, density=True)
    
    ax.set_xlabel(feature, fontsize=12, fontweight='bold')
    ax.set_ylabel('Densidade', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribuição de {feature}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\dados1\Documents\tcc\tcc\fig13_distribuicao_top_variaveis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

statistical_tests = []

for col in ['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)]:
    legit_values = df[df['Class']==0][col]
    fraud_values = df[df['Class']==1][col]
    
    statistic, p_value = stats.mannwhitneyu(legit_values, fraud_values)
    
    statistical_tests.append({
        'Variável': col,
        'Estatística U': statistic,
        'p-valor': p_value,
        'Significativo (α=0.05)': 'Sim' if p_value < 0.05 else 'Não'
    })

df_stats_tests = pd.DataFrame(statistical_tests)
df_stats_tests.to_csv(r'C:\Users\dados1\Documents\tcc\tcc\testes_estatisticos.csv', 
                      index=False, encoding='utf-8-sig')