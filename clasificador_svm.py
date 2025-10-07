# Nombre del archivo: clasificador_svm.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier (SVC)
from sklearn.metrics import accuracy_score

# 1. Cargar el Dataset y dividir en características (X) y etiquetas (y)
iris = load_iris()
X, y = iris.data, iris.target 

# 2. División de Datos (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y 
)

# 3. Inicializar y Entrenar el Modelo SVM
# Usaremos el kernel 'linear' (lineal), el más simple, para encontrar un hiperplano recto.
# El hiperplano maximiza el margen entre las clases.
svm_modelo = SVC(kernel='linear', random_state=42)
print("Entrenando clasificador SVM con kernel 'linear'...")
svm_modelo.fit(X_train, y_train)

# 4. Realizar Predicciones
y_pred = svm_modelo.predict(X_test)

# 5. Evaluación del Modelo
accuracy = accuracy_score(y_test, y_pred) * 100

print("-" * 40)
print(f"Precisión del Modelo SVM (Kernel Lineal): {accuracy:.2f}%")
print("-" * 40)

# 6. Identificar los Vectores de Soporte (Los datos cruciales)
# Los vectores de soporte son los puntos de datos más cercanos al hiperplano, que definen el margen.
num_sv = svm_modelo.support_vectors_.shape[0]
print(f"Número de Vectores de Soporte encontrados: {num_sv}")