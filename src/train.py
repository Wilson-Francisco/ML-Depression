# Importando as biblitecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

# visualizar o shape do dataframe
df.shape

# Separar as minhas variaveis
X, y = df.drop(["Depression", "id", 'Profession', 'Work Pressure', 'Job Satisfaction'], axis=1), df["Depression"]

# Analise exploratória dos dados

# Criando faixa etária para 'Age'
bins = [0, 18, 24, 30, 40, 50, 59]
labels = ['Até 18 anos', 'De 19 até 24 anos', 'De 25 até 30 anos', 'De 31 até 40 anos', 'De 41 até 50 anos', 'acima de 50 anos']
X['Age_Range'] = pd.cut(X['Age'], bins=bins, labels=labels)
X.groupby(['Age_Range'], observed=True).size().astype("category")

