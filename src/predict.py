import pandas as pd
import re
import numpy as np
import mlflow
import mlflow.sklearn


# Carregando os dados
df = pd.read_csv("../data/student_depression_dataset.csv")

# Criar função para extrair horas da variável 'Sleep Duration
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

# Import do modelo do mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model = mlflow.sklearn.load_model("models:/model_student_depression/2")
print(model)

# Import das features do mlflow
features = model.feature_names_in_

# Simulando dados novos
amostra = df[df["Sleep Duration"] == df["Sleep Duration"].max()].sample(18)
amostra = amostra.drop("Depression", axis=1)
#
# # Predição do modelo
# predicao = model.predict_proba(amostra[features])[:,1]
# amostra["Proba"] = predicao
#
# print(amostra)