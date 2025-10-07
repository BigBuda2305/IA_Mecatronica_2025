# Nombre del archivo: regresion_lineal.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el Dataset y seleccionar UNA característica (Regresión Simple)
diabetes = datasets.load_diabetes()

# Seleccionamos la segunda característica (índice 1) para el eje X
X = diabetes.data[:, np.newaxis, 1] 
y = diabetes.target # La variable a predecir

# 2. División de Datos (20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Creación y Entrenamiento del Modelo
modelo_regresion = linear_model.LinearRegression()
modelo_regresion.fit(X_train, y_train) # ¡El corazón del ML: el modelo aprende!

# 4. Realizar Predicciones
y_pred = modelo_regresion.predict(X_test) # Predecir valores para los datos no vistos

# 5. Evaluación del Modelo (Métricas clave)
# Coeficiente: Es la pendiente (m) de la línea (y = mx + b). Indica la relación.
print(f"Coeficiente (pendiente): {modelo_regresion.coef_[0]:.2f}")
# Error Cuadrático Medio (MSE): Mide el error promedio al cuadrado de las predicciones. (Menor es mejor)
print(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
# R2 Score: Indica qué tan bien los datos se ajustan a la línea. (Cercano a 1.0 es mejor)
print(f"R2 Score (Bondad del Ajuste): {r2_score(y_test, y_pred):.2f}")

# 6. Visualización de la Regresión (Crucial para entender el modelo)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Datos Reales (Prueba)') # Puntos reales
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Línea de Regresión') # La línea que el modelo "aprendió"
plt.title('Regresión Lineal Simple (Predicción vs. Característica)')
plt.xlabel('Característica de Entrada')
plt.ylabel('Valor a Predecir')
plt.legend()
plt.show()