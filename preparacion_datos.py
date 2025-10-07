# Nombre del archivo: preparacion_datos.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Cargar el Dataset (Simulando la lectura de datos de sensores)
iris = load_iris()
X, y = iris.data, iris.target 

print("--- 1. Vista Previa de los Datos ---")
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head()) # Muestra las primeras 5 filas
print(f"\nNúmero total de muestras (filas): {X.shape[0]}")
print("-" * 35)

# 2. División de Datos (La función train_test_split es central en ML)
# 70% para aprender (train), 30% para evaluar (test).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, # 30% para el conjunto de prueba
    random_state=42, # Asegura que la división sea reproducible
    stratify=y # Asegura que las clases se distribuyan uniformemente
)

# 3. Verificación de las Divisiones
print("--- 2. Tamaños después de la división 70/30 ---")
print(f"Tamaño de Entrenamiento (X_train): {X_train.shape}")
print(f"Tamaño de Prueba (X_test): {X_test.shape}")
print(f"Número de etiquetas de entrenamiento (y_train): {y_train.shape[0]}")
print(f"Número de etiquetas de prueba (y_test): {y_test.shape[0]}")