import pandas as pd
import seaborn as sns
from sklearn import datasets
import streamlit as st

# Carregar o dataset Iris
df = sns.load_dataset('iris')

# Exibir informações iniciais no Streamlit
st.title('Análise de Conjunto de Dados Iris')
st.write(df.head())
st.subheader('Estatisticas Descritivas')
st.write(df.describe())

#Graficos interativos
st.subheader('Gráfico de Dispersão: Sepal Lenght vc Sepal Width')
st.write('Visualização de dispersão entre as variáveis Sepal Length e Sepal Width')
scatter_plot = sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
st.pyplot(scatter_plot.figure)

st.subheader('Distribuição do comprimento da pétala')
st.write('Distribuição do comprimento da pétala por espécie')
hist_plot = sns.histplot(data=df, x='petal_length', hue='species', kde=True)
st.pyplot(hist_plot.figure)