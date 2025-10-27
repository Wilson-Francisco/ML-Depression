import re

import mlflow.sklearn
import numpy as np
import pandas as pd

# Carregando os dados
df = pd.read_csv("../data/student_depression_dataset.csv")

# Criar função para extrair horas da variável 'Sleep Duration'
def process_sleep_duration(series):
    def extract_hours(s):
        match = re.search(r"(\d+(\.\d+)?)", str(s))
        return float(match.group(1)) if match else np.nan
    series = series.apply(extract_hours)
    if series.isna().sum() == 0:
        print("Não há valores nulos.")
    else:
        series = series.fillna(int(series.mean()))
        print("Valores numéricos extraídos, preenchidos com média.")
    return series.values.reshape(-1, 1)

# Capturar sempre a versão mais recente do  modelo no mlfow
models = mlflow.search_registered_models(filter_string="name = 'model_student_depression' ")
latest_versions = max([i.version for in models[0].latest_versions[]])

# Import do modelo do mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model = mlflow.sklearn.load_model(f"models:/model_student_depression/{latest_versions}")
print(model)

# Import das features do mlflow
features = model.feature_names_in_

# Simulando dados novos
amostra = df[df["CGPA"] < df["CGPA"].max()].sample(30)
amostra = amostra.drop("Depression", axis=1)

# Aplicando a extração da variável 'Sleep Duration' no dataframe da amostra
amostra['Sleep Duration'] = process_sleep_duration(amostra['Sleep Duration'])

# Predição do modelo
predicao = model.predict_proba(amostra[features])[:,1]
amostra["Proba_Depression"] = predicao

print(amostra)