# Nombre del archivo: clasificador_knn.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Cargar el Dataset y dividir en características (X) y etiquetas (y)
iris = load_iris()
X, y = iris.data, iris.target 

# 2. División de Datos (70% Train, 30% Test) - Reutilizando la lógica del Día 8
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y 
)

# 3. Inicializar y Entrenar el Modelo k-NN
# El modelo k-NN es un "Lazy Learner" (aprendizaje perezoso): el entrenamiento es rápido,
# solo almacena los datos de entrenamiento.
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
print(f"Entrenando clasificador k-NN con k = {k}...")
knn.fit(X_train, y_train)

# 4. Realizar Predicciones
y_pred = knn.predict(X_test)

# 5. Evaluación del Modelo
# La Precisión (Accuracy) es la métrica más simple: (predicciones correctas / total de predicciones)
accuracy = accuracy_score(y_test, y_pred) * 100

print("-" * 40)
print(f"Precisión del Modelo k-NN (k={k}): {accuracy:.2f}%")
print("-" * 40)

# 6. Prueba con un Nuevo Dato (Simulando un nuevo sensor)
# Una muestra de datos de sensor nunca antes vistos: [long. sépalo, ancho sépalo, long. pétalo, ancho pétalo]
nuevo_dato = np.array([[5.0, 3.0, 1.5, 0.2]]) 
prediccion_nueva = knn.predict(nuevo_dato)

# Mapear el resultado numérico (0, 1, o 2) a la etiqueta de la flor
especie_predicha = iris.target_names[prediccion_nueva][0]

print(f"Nuevo dato a clasificar: {nuevo_dato[0]}")
print(f"Clase predicha (Especie): {especie_predicha}")