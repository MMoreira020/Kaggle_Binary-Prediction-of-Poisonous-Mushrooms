# Kaggle Playground Series S4E8 - Predição Binária de Cogumelos Venenosos

Este repositório contém o código e a análise para a competição **Kaggle Playground Series Season 4 Episode 8**: [Predição Binária de Cogumelos Venenosos](https://www.kaggle.com/competitions/playground-series-s4e8/overview).  
O objetivo desta competição é construir um modelo que consiga prever com precisão se um cogumelo é **venenoso (p)** ou **comestível (e)** com base em diversas características, como formato do chapéu, cor, características das lamelas e propriedades do caule.

---

## Conteúdo do Notebook

O notebook [predicting_poisonous_mushrooms.ipynb](predicting_poisonous_mushrooms.ipynb) percorre as seguintes etapas:

1. **Carregamento e Exploração dos Dados:**
    - Carregamento dos arquivos `train.csv` e `test.csv` com o pandas.
    - Exibição da forma dos conjuntos de dados e das primeiras linhas do conjunto de treino.

2. **Tratamento de Valores Ausentes:**
    - Identificação das colunas com valores ausentes e contagem dessas ocorrências.
    - Implementação de uma estratégia para lidar com os dados ausentes:
        - Remoção de linhas com poucos valores ausentes.
        - Preenchimento dos valores ausentes com a string `"missing"` em colunas com muitos valores ausentes.

3. **Análise de Frequência de Valores:**
    - Definição de uma função para calcular e exibir a contagem e porcentagem de valores únicos por coluna.
    - Aplicação dessa função no conjunto de treino para entender a distribuição das variáveis.

4. **Filtragem de Categorias Raras:**
    - Identificação e remoção de linhas com categorias raras (ocorrências menores que 200) em cada coluna de atributos (exceto ‘id’ e ‘class’) para reduzir ruído e melhorar a generalização do modelo.

5. **Engenharia e Codificação de Variáveis:**
    - Separação da variável-alvo (‘class’) das variáveis preditoras.
    - Codificação one-hot das variáveis categóricas para transformá-las em formato numérico adequado aos algoritmos de machine learning.

6. **Pré-processamento dos Dados de Teste:**
    - Aplicação dos mesmos passos de tratamento (valores ausentes e codificação one-hot) ao arquivo `test.csv` para garantir consistência com o conjunto de treino.
    - Alinhamento das colunas do conjunto de teste com o de treino após a codificação one-hot, tratando possíveis discrepâncias.

7. **Análise de Correlação e Seleção de Variáveis:**
    - Cálculo da matriz de correlação para os atributos numéricos.
    - Geração de um **mapa de calor (heatmap)** das correlações para identificar visualmente relações entre variáveis.
    - Salvamento do heatmap como imagem (`mapa_correlacao.png`) para referência.
    - Identificação de **pares de atributos altamente correlacionados** (correlação de Pearson `r > 0.85`) para considerar remoção ou análise de colinearidade.
    - Exemplo: `cap-diameter` e `stem-width` apresentaram alta correlação (`r = 0.83`), sugerindo atenção em modelos lineares.

    ![Mapa de Correlação](mapa_correlacao.png)

8. **Divisão dos Dados em Conjuntos de Treinamento e Validação:**
    - Utilização da função `train_test_split` do `sklearn.model_selection` para dividir o conjunto de treino em treino e validação.
    - Uso de divisão estratificada para manter a distribuição das classes nos dois subconjuntos.

9. **Padronização dos Dados:**
    - Aplicação do `StandardScaler` do `sklearn.preprocessing` para padronizar as variáveis numéricas, melhorando o desempenho e a convergência dos modelos.

10. **Treinamento e Avaliação dos Modelos:**
    - Definição das funções `train_model` e `evaluate_model` para facilitar o treinamento e avaliação de diferentes classificadores.
    - Treinamento e avaliação de três modelos de machine learning:
        - **Regressão Logística:** Modelo linear usado como baseline.
        - **Random Forest Classifier:** Método de ensemble conhecido pela robustez em dados tabulares.
        - **XGBoost Classifier:** Algoritmo de boosting gradiente que frequentemente atinge desempenho de ponta em tarefas de classificação.
    - Avaliação de desempenho com métricas como Acurácia, F1-Score, Matriz de Confusão e Curva AUC-ROC.

11. **Predição e Geração do Arquivo de Submissão:**
    - Padronização do conjunto de teste com o `StandardScaler` ajustado no treino.
    - Uso do modelo XGBoost treinado (melhor desempenho) para prever as classes no conjunto de teste.
    - Decodificação das previsões numéricas para os rótulos originais ('e' ou 'p').
    - Criação de um arquivo CSV pronto para submissão (`predictions_xgb.csv`) com as colunas 'id' e 'class'.

---

## Modelos e Desempenho

Os seguintes modelos foram treinados e avaliados:

| Modelo                    | Acurácia | F1-Score | AUC-ROC |
|---------------------------|----------|----------|---------|
| Regressão Logística       | 0.8897   | 0.8897   | 0.9373  |
| Random Forest Classifier  | 0.9920   | 0.9920   | 0.9966  |
| XGBoost Classifier        | 0.9924   | 0.9924   | 0.9968  |

O **XGBoost Classifier** obteve o melhor desempenho no conjunto de validação com acurácia de 0.9924 e AUC-ROC de 0.9968. Esse modelo foi utilizado para gerar as previsões finais no conjunto de teste.

---

## Observações Adicionais

- A análise de correlação e a filtragem de variáveis foram úteis para entender melhor a estrutura do dataset e orientar decisões no design do modelo.
- O heatmap gerado (`mapa_correlacao.png`) pode ser útil em futuras etapas de seleção de atributos, especialmente se forem aplicadas técnicas de redução de dimensionalidade ou regularização.

---
