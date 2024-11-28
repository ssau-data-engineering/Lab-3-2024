from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Путь к папке с данными
data_dir = '/data/'  # Папка, где находится датасет
output_dir = '/data/'  # Папка для сохранения обработанных данных
os.makedirs(output_dir, exist_ok=True)  # Создаем папку, если её нет

def load_and_preprocess_data():
    # Загружаем ирисовый датасет
    file_path = os.path.join(data_dir, 'iris.csv')  # Убедитесь, что файл называется именно так
    df = pd.read_csv(file_path)

    # Кодируем категориальные значения меток (variety)
    label_encoder = LabelEncoder()
    df['variety'] = label_encoder.fit_transform(df['variety'])

    # Разделяем данные на X и y
    X = df.drop(columns=['variety'])  # Признаки
    y = df['variety']                  # Метки

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Разделяем тестовую выборку на валидационную и тестовую
    x_val, x_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=163)

    # Сохраняем обучающие, тестовые и валидационные наборы данных
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    x_val.to_csv(os.path.join(output_dir, 'x_val.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)  
    
    print("Предобработанные данные сохранены в папке:", output_dir)

if __name__ == "__main__":
    load_and_preprocess_data()
