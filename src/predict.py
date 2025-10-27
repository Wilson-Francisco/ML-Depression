import pandas as pd
import mlflow
import mlflow.sklearn






# Import do modelo do mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model = mlflow.sklearn.log_model("models:/model_student_depression/1")
print(model)

# Import das features do mlflow
features = model.feature_names_in_

# Simulando dados novos
amostra = df[df["Sleep Duration"] == df["Sleep Duration"].max()].sample(60)