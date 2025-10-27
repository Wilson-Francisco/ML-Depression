# Importando as biblitecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import scipy.stats as stats

from scipy.stats import chi2_contingency

from sklearn import pipeline

from sklearn import tree

from feature_engine.encoding import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics


# Importar os dados
df = pd.read_csv("../data/student_depression_dataset.csv")

# Visualizar os dados
df.head()

# Visualizar o shape do dataframe
df.shape

# Separar as minhas variáveis
X, y = df.drop(["Depression", "id", 'Profession', 'Work Pressure', 'Job Satisfaction'], axis=1), df["Depression"]

# Analise exploratória dos dados

# Verificando o dataframe
X.info()

# Criando faixa etária para 'Age'
bins = [0, 18, 24, 30, 40, 50, 59]
labels = ['Até 18 anos', 'De 19 até 24 anos', 'De 25 até 30 anos', 'De 31 até 40 anos', 'De 41 até 50 anos', 'acima de 50 anos']
X['Age_Range'] = pd.cut(X['Age'], bins=bins, labels=labels)
X.groupby(['Age_Range'], observed=True).size().astype("category")

# Verificando null na variável 'Age_Range'
X['Age_Range'].isna().sum()

# Alterando  variáveis do tipo objeto para numérica
X["Academic Pressure"].astype("float64")
X['Financial Stress'] = pd.to_numeric(X['Financial Stress'], errors='coerce')

# Preenchendo os valores null
X['Financial Stress'] = X['Financial Stress'].fillna(int(X['Financial Stress'].mean()))

# Verificando o dataframe após o tratamento de algumas variáveis
X.info()

# Extrair as horas da variável 'Sleep Duration'
def extrar_horas(s):
    match = re.search(r"(\d+(\.\d+)?)", str(s))
    return float(match.group(1)) if match else np.nan
X['Sleep Duration'] = X['Sleep Duration'].apply(extrar_horas)

print(X['Sleep Duration'].isnull().sum())

# Preenchendo os valores null
X['Sleep Duration'] = X['Sleep Duration'].fillna(int(X['Sleep Duration'].mean()))

# Eliminando a coluna "Age" e criando um novo dataframe
df_analise = X.drop(["Age"], axis=1).copy()
df_analise["Depression"] = y

# Visualizar os dados do novo dataframe
print(df_analise.head())

# Total de valores únicos de cada variável
valores_unicos = []
for i in df_analise.columns[0:14].tolist():
    print(i, ':', len(df_analise[i].astype(str).value_counts()))
    valores_unicos.append(len(df_analise[i].astype(str).value_counts()))

# Visualizando algumas medidas estatísticas.
df_analise.describe()
print(df_analise.describe())

# Separando as variáveis categóricas
variaveis_categoricas = []
for i in df_analise.columns[0:14].tolist():
        if df_analise.dtypes[i] == 'object' or df_analise.dtypes[i] == 'category':
            variaveis_categoricas.append(i)

# Separar as features e target
features = df_analise.columns[0:-1]
target = "Depression"

# Visualizar os gráficos das variáveis em função de 'Depression'
# Ajustar o tamanho dos gráficos
plt.rcParams["figure.figsize"] = [10.00, 4.00]
plt.rcParams["figure.autolayout"] = True

colunas = ['Gender', 'Academic Pressure','Study Satisfaction', 'Sleep Duration',
       'Dietary Habits', 'Have you ever had suicidal thoughts ?',
       'Work/Study Hours', 'Financial Stress',
       'Family History of Mental Illness', 'Age_Range']
for i in colunas:
    sns.countplot(data = df_analise, x = df_analise[i], hue = "Depression")
    plt.show()

# Verificando o teste estatístico da correlação das variáveis categóricas
# Ho não há evidências de associação entre variaveis_categoricas e Depression
def teste_hipotese_categorica(df_analise, variaveis_categoricas):
    contingency = pd.crosstab(df_analise[variaveis_categoricas], df_analise['Depression'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nTeste de Independência: {variaveis_categoricas} x Depression")
    print(contingency)
    print(f"Chi2: {chi2:.2f}")
    print(f"p-valor: {p:.4f}")
    if p < 0.05:
        print(f"Resultado: Rejeitamos Ho, há associação entre {variaveis_categoricas} e Depression.")
    else:
        print(f"Resultado: Não rejeitamos Ho, não há evidências de associação entre {variaveis_categoricas} e Depression.")

for var in variaveis_categoricas:
    teste_hipotese_categorica(df_analise, var)

# Tabela de correspondência de variáveis
df_analise[["City", "Depression"]].groupby(["City"], as_index=False).sum()
df_analise[["Degree", "Depression"]].groupby(["Degree"], as_index=False).sum()
print(df_analise[["City", "Depression"]].groupby(["City"], as_index=False).sum())
print(df_analise[["Degree", "Depression"]].groupby(["Degree"], as_index=False).sum())

# Cria o encoder e aplica OneHotEncoder
onehot = OneHotEncoder(variables = variaveis_categoricas)

# Dividir os dados em treino e teste para iniciar a fase de criação do modelo
X_train, X_test, y_train, y_test = train_test_split(df_analise[features], df_analise[target] , test_size = 0.2, random_state = 42)

# Modelo de árvore de Classificador de Árvore de Decisão
clf_tree = tree.DecisionTreeClassifier(max_depth=5, random_state = 42)

# Normalizar as variáveis
norm = MinMaxScaler()

# Pipeline com todos objetos
model_pipeline = pipeline.Pipeline(steps = [("onehot", onehot),
                                            ("norm ", norm ),
                                            ("clf_tree", clf_tree)])

# Ajustando o modelo
model_pipeline.fit(X_train[features], y_train)

# Salvando o algoritmo
model = pd.Series(
    {
        "model": model_pipeline,
        "features": features
    } )

# Métricas de treino do modelo
pred_train = model["model"].predict(X_train[features])
pred_proba_train = model["model"].predict_proba(X_train[features])[:,1]

# Calcular acuracia do modelo de treino
scores_train = model["model"].score(X_train[features], y_train)
print(scores_train)

# Calcular curva ROC do modelo de treino
scores_roc_auc_train = metrics.roc_auc_score(y_train, pred_proba_train)
print(scores_roc_auc_train)

# Métricas de teste do modelo
pred_test = model["model"].predict(X_test[features])
pred_proba_test = model["model"].predict_proba(X_test[features])[:,1]

# Calcular acuracia do modelo de teste
scores_test = model["model"].score(X_test[features], y_test)
print(scores_test)

# Calcular curva ROC do modelo de teste
scores_auc_test = metrics.roc_auc_score(y_test, pred_proba_test)
print(scores_auc_test)

# Treino da Curva ROC
roc_curve_train = metrics.roc_curve(y_train, pred_proba_train)

# Teste da Curva ROC
roc_curve_test = metrics.roc_curve(y_test, pred_proba_test)

# Gráfico da curva ROC
plt.plot(roc_curve_train[0], roc_curve_train[1])
plt.plot(roc_curve_test[0], roc_curve_test[1])
plt.grid(True)
plt.plot([0,1],[0,1], "--", color="black")
plt.title("Curva ROC")
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.legend(
[
    f"Treino: {100*scores_train:.2f}%",
    f"Teste: {100*scores_test:.2f}%"
])
plt.show()