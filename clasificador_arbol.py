# Nombre del archivo: clasificador_arbol.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd # ¡AGREGA ESTA LÍNEA!

# 1. Cargar y Dividir los Datos
iris = load_iris()
X, y = iris.data, iris.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y 
)

# --- MODELO 1: ÁRBOL DE DECISIÓN (Decision Tree) ---
# Un solo árbol que toma decisiones secuenciales. Tiende al sobreajuste.
arbol_modelo = DecisionTreeClassifier(random_state=42)
arbol_modelo.fit(X_train, y_train)

# Predicción y Precisión
y_pred_arbol = arbol_modelo.predict(X_test)
accuracy_arbol = accuracy_score(y_test, y_pred_arbol) * 100
print("-" * 40)
print(f"Precisión del Árbol de Decisión: {accuracy_arbol:.2f}%")
print("-" * 40)


# --- MODELO 2: RANDOM FOREST (Bosque Aleatorio) ---
# Un conjunto (ensemble) de muchos árboles pequeños, mejorando la precisión y reduciendo el sobreajuste.
# n_estimators=100 significa que usa 100 árboles.
forest_modelo = RandomForestClassifier(n_estimators=100, random_state=42)
forest_modelo.fit(X_train, y_train)

# Predicción y Precisión
y_pred_forest = forest_modelo.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest) * 100
print(f"Precisión del Random Forest (100 árboles): {accuracy_forest:.2f}%")
print("-" * 40)

# 3. Visualizar la Importancia de las Características (Feature Importance)
# Los Random Forests nos dicen qué características fueron más importantes para la decisión.
feature_importances = pd.Series(forest_modelo.feature_importances_, index=iris.feature_names)

plt.figure(figsize=(8, 5))
feature_importances.nlargest(4).plot(kind='barh')
plt.title('Importancia de Características (Random Forest)')
plt.xlabel('Importancia Relativa')
plt.show() # Muestra el gráfico en tu Mac