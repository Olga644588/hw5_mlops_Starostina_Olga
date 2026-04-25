import pandas as pd

# Загрузка данных
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

# Вывод имён столбцов
print("Столбцы в train.csv:")
print(train_data.columns.tolist())

print("\nСтолбцы в test.csv:")
print(test_data.columns.tolist())

# Вывод первых 5 строк для наглядности
print("\nПервые 5 строк train.csv:")
print(train_data.head())
