import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_excel('base_serasa_figura_1.xlsx', sheet_name='basetcc')

# Variação Percentual em relação a 2024
variacao_percentual = [41.6, 37.4, -2.0]

meses_desejados = ['2025-01-01', '2025-02-01', '2025-03-01']
df_filtrado = df[df['Mês'].isin(meses_desejados)].copy()

df_filtrado['Mês'] = pd.to_datetime(df_filtrado['Mês'])
df_filtrado.sort_values('Mês', inplace=True)
df_filtrado = df_filtrado.reset_index(drop=True)

sns.set_theme(style="white")
plt.figure(figsize=(9,6))

barplot = sns.barplot(
    data=df_filtrado,
    x="Mês",
    y="Total",
    color="#1f77b4",
    edgecolor="black",
    linewidth=1.2
)

plt.xlabel("Mês", fontsize=13, labelpad=10)
plt.ylabel("Total de Tentativas", fontsize=13, labelpad=10)

plt.xticks(
    ticks=range(len(df_filtrado)),
    labels=[mes.strftime("%b/%Y") for mes in df_filtrado["Mês"]],
    fontsize=11
)

for i, row in df_filtrado.iterrows():
    barplot.text(
        i, 
        row['Total'] + (row['Total'] * 0.02), 
        f"{row['Total']:,.0f}", 
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )
    
    sinal = "+" if variacao_percentual[i] >= 0 else ""
    texto_variacao = f"{sinal}{variacao_percentual[i]:.1f}% vs 2024"
    
    barplot.text(
        i,
        row['Total'] * 0.5, 
        texto_variacao,
        ha='center',
        va='center',
        fontsize=11,
        fontweight='bold',
        color='white',
        bbox=dict(facecolor='#003366', alpha=0.85, boxstyle='round,pad=0.4')
    )

sns.despine()
plt.tight_layout()
plt.show()