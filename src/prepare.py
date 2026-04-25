import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Загружаем параметры
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Читаем сырые данные
df = pd.read_csv('data\\raw\\iris.csv')

# Разделяем на train/test
train, test = train_test_split(
    df,
    test_size=params['split_ratio'],
    random_state=params['random_state']
)

# Сохраняем обработанные данные
train.to_csv('data\\processed\\train.csv', index=False)
test.to_csv('data\\processed\\test.csv', index=False)

print("Данные успешно обработаны и сохранены в data/processed/")
