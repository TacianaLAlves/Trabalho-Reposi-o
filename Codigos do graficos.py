import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from sklearn.datasets import load_iris, fetch_california_housing
from datetime import datetime

pasta_graficos = "graficos_salvos"
os.makedirs(pasta_graficos, exist_ok=True)

print(f"📁 Pasta criada: {os.path.abspath(pasta_graficos)}")

# Carregar datasets
iris = sns.load_dataset('iris')
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['PRICE'] = housing.target * 100000

# Função para salvar gráficos
def salvar_grafico(nome_arquivo, dpi=300):
    """Salva o gráfico atual na pasta de gráficos"""
    caminho_completo = os.path.join(pasta_graficos, nome_arquivo)
    plt.savefig(caminho_completo, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"💾 Gráfico salvo: {caminho_completo}")
    return caminho_completo

def salvar_plotnine(grafico, nome_arquivo, dpi=300, width=10, height=6):
    """Salva gráfico plotnine na pasta de gráficos"""
    caminho_completo = os.path.join(pasta_graficos, nome_arquivo)
    ggsave(grafico, caminho_completo, dpi=dpi, width=width, height=height)
    print(f"💾 Gráfico plotnine salvo: {caminho_completo}")
    return caminho_completo

#| label: fig-iris-scatter-save
#| fig-cap: "Relação entre comprimento e largura das pétalas por espécie"

# Criar figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Sepal
cores = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for especie in iris['species'].unique():
    dados_especie = iris[iris['species'] == especie]
    ax1.scatter(dados_especie['sepal_length'], dados_especie['sepal_width'],
               label=especie, alpha=0.7, s=60, c=cores[especie])

ax1.set_xlabel('Comprimento da Sépala (cm)')
ax1.set_ylabel('Largura da Sépala (cm)')
ax1.set_title('Relação Sépala - por Espécie')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Petal
for especie in iris['species'].unique():
    dados_especie = iris[iris['species'] == especie]
    ax2.scatter(dados_especie['petal_length'], dados_especie['petal_width'],
               label=especie, alpha=0.7, s=60, c=cores[especie])

ax2.set_xlabel('Comprimento da Pétala (cm)')
ax2.set_ylabel('Largura da Pétala (cm)')
ax2.set_title('Relação Pétala - por Espécie')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Salvar gráfico
salvar_grafico("scatter_iris_sepal_petal.png")
plt.show()

#| label: fig-housing-dist-save
#| fig-cap: "Distribuição de preços e relação com renda"

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Histograma de preços
ax[0,0].hist(housing_df['PRICE'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax[0,0].set_xlabel('Preço da Casa (US$)')
ax[0,0].set_ylabel('Frequência')
ax[0,0].set_title('Distribuição de Preços de Casas')
ax[0,0].grid(True, alpha=0.3)

# Boxplot por faixa de renda
housing_df['Renda_Categoria'] = pd.cut(housing_df['MedInc'], bins=5)
ax[0,1].boxplot([housing_df[housing_df['Renda_Categoria'] == cat]['PRICE'] 
                for cat in housing_df['Renda_Categoria'].cat.categories],
               labels=[f'Cat {i+1}' for i in range(5)])
ax[0,1].set_xlabel('Categoria de Renda')
ax[0,1].set_ylabel('Preço da Casa (US$)')
ax[0,1].set_title('Preços por Categoria de Renda')
ax[0,1].grid(True, alpha=0.3)

# Scatter plot: Renda vs Preço
ax[1,0].scatter(housing_df['MedInc'], housing_df['PRICE'], alpha=0.5, s=20, color='purple')
ax[1,0].set_xlabel('Renda Média (MedInc)')
ax[1,0].set_ylabel('Preço da Casa (US$)')
ax[1,0].set_title('Relação: Renda vs Preço')
ax[1,0].grid(True, alpha=0.3)

# Gráfico de densidade
housing_df['PRICE'].plot.kde(ax=ax[1,1], color='red', linewidth=2)
ax[1,1].set_xlabel('Preço da Casa (US$)')
ax[1,1].set_ylabel('Densidade')
ax[1,1].set_title('Densidade de Preços')
ax[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# Salvar gráfico
salvar_grafico("housing_analise_completa.png")
plt.show()

#| label: fig-plotnine-scatter-save
#| fig-cap: "Relação entre medidas das flores com faceting"

# Criar gráfico
grafico_iris = (ggplot(iris)
 + aes(x='petal_length', y='petal_width', color='species')
 + geom_point(size=3, alpha=0.7)
 + facet_wrap('~ species', ncol=3)
 + labs(
     x='Comprimento da Pétala (cm)',
     y='Largura da Pétala (cm)',
     title='Relação Pétala por Espécie',
     color='Espécie'
 )
 + theme_minimal()
 + theme(
     figure_size=(12, 5),
     axis_text_x=element_text(angle=0),
     legend_position='right'
 )
)

# Salvar e exibir
salvar_plotnine(grafico_iris, "plotnine_scatter_iris.png")
grafico_iris

#| label: fig-plotnine-housing-save
#| fig-cap: "Análise multivariada de preços de casas"

# Preparar dados
housing_plot = housing_df.copy()
housing_plot['Renda_Cat'] = pd.cut(housing_plot['MedInc'], 
                                  bins=[0, 2, 4, 6, 8, 10], 
                                  labels=['0-2', '2-4', '4-6', '6-8', '8+'])

# Criar gráfico
grafico_housing = (ggplot(housing_plot)
 + aes(x='Renda_Cat', y='PRICE', fill='Renda_Cat')
 + geom_violin(alpha=0.7, show_legend=False)
 + geom_boxplot(width=0.2, alpha=0.8, show_legend=False)
 + labs(
     x='Categoria de Renda',
     y='Preço da Casa (US$)',
     title='Distribuição de Preços por Categoria de Renda',
     subtitle='Violin plot com boxplot overlay'
 )
 + scale_y_continuous(labels=lambda x: [f'${int(val/1000)}k' for val in x])
 + theme_minimal()
 + theme(
     figure_size=(10, 6),
     axis_text_x=element_text(angle=45),
     plot_title=element_text(size=14, weight='bold'),
     plot_subtitle=element_text(size=10)
 )
)

# Salvar e exibir
salvar_plotnine(grafico_housing, "plotnine_violin_housing.png")
grafico_housing
