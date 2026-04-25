import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Инициализация MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris Classification")

# Загрузка подготовленных данных
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

# Используем правильное имя целевого столбца
X_train = train_data.drop("variety", axis=1)
y_train = train_data["variety"]
X_test = test_data.drop("variety", axis=1)
y_test = test_data["variety"]

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание и расчёт точности
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Логирование в MLflow
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    
    # Сохранение и логирование модели
    joblib.dump(model, "models\\model.pkl")
    mlflow.log_artifact("models\\model.pkl")

print(f"Точность модели: {acc:.3f}")
