# Nombre del archivo: evaluacion_modelo.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns # Necesario para la Matriz de Confusión

# 1. Cargar el Modelo y los Datos (k-NN)
iris = load_iris()
X, y = iris.data, iris.target 
target_names = iris.target_names # Nombres de las clases para la Matriz

# División de Datos (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y 
)

# Entrenar el modelo k-NN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 2. Matriz de Confusión (El Corazón de la Evaluación)
# Muestra dónde se equivocó el modelo (valores reales vs. valores predichos)
cm = confusion_matrix(y_test, y_pred)

print("--- 1. Matriz de Confusión Numérica ---")
print(cm)
print("\n")

# 3. Reporte de Clasificación (Precision, Recall, F1-Score)
# Precision: De todas las veces que predijo una clase, ¿cuántas fueron correctas?
# Recall (Exhaustividad): De todas las muestras REALES de una clase, ¿cuántas predijo correctamente?
# F1-Score: Es el promedio armónico de Precision y Recall (un buen indicador general).
print("--- 2. Reporte Completo de Clasificación (Precisión, Recall, F1) ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# 4. Visualización de la Matriz de Confusión
# Esto facilita la interpretación de los errores
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.title('Matriz de Confusión (k-NN)')
plt.show() # Muestra el gráfico en tu Mac