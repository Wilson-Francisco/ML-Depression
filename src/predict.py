import pandas as pd
import mlflow
import mlflow.sklearn


# Carregando os dados
df = pd.read_csv("../data/student_depression_dataset.csv")


# Criando faixa etária para 'Age'
bins = [0, 18, 24, 30, 40, 50, 59]
labels = ['Até 18 anos', 'De 19 até 24 anos', 'De 25 até 30 anos', 'De 31 até 40 anos', 'De 41 até 50 anos', 'acima de 50 anos']
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels)
df.groupby(['Age'], observed=True).size().astype("category")

# Import do modelo do mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model = mlflow.sklearn.load_model("models:/model_student_depression/1")
print(model)

# Import das features do mlflow
features = model.feature_names_in_

# Simulando dados novos
amostra = df[df["Sleep Duration"] == df["Sleep Duration"].max()].sample(18)
amostra = amostra.drop("Depression", axis=1)

# Predição do modelo
predicao = model.predict_proba(amostra[features])[:,1]
amostra["Proba"] = predicao