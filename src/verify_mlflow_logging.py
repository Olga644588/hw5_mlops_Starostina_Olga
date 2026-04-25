import mlflow
from mlflow.tracking import MlflowClient

# Явно указываем URI отслеживания
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Получение ID эксперимента
experiment = client.get_experiment_by_name("Iris Classification")
experiment_id = experiment.experiment_id

# Поиск последнего запуска
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)
run = runs[0]
run_id = run.info.run_id

print(f"ID запуска: {run_id}")
print("\n--- ПРОВЕРКА ЛОГИРОВАНИЯ ---")

# Проверка параметров
print("\n1. Параметры (Parameters):")
params = run.data.params
if "model" in params:
    print(f"✓ Параметр 'model' найден: {params['model']}")
else:
    print("✗ Параметр 'model' не найден!")

# Проверка метрик
print("\n2. Метрики (Metrics):")
metrics = run.data.metrics
if "accuracy" in metrics:
    print(f"✓ Метрика 'accuracy' найдена: {metrics['accuracy']:.3f}")
else:
    print("✗ Метрика 'accuracy' не найдена!")

# Проверка артефактов
print("\n3. Артефакты (Artifacts):")
artifacts = client.list_artifacts(run_id)
artifact_names = [artifact.path for artifact in artifacts]
if "model.pkl" in artifact_names:
    print("✓ Артефакт 'model.pkl' найден")
else:
    print("✗ Артефакт 'model.pkl' не найден!")

# Дополнительная информация
print(f"\n4. Дополнительная информация:")
print(f"   Эксперимент: {experiment.name}")
print(f"   Время запуска: {run.info.start_time}")
print(f"   Статус: {run.info.status}")
