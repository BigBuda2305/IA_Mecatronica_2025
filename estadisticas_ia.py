import numpy as np
import pandas as pd

# Datos simulados de Eficiencia (en porcentaje) de un sistema robótico
# Notarás que un par de valores están lejos de la media (posibles fallos o "outliers")
eficiencia = np.array([85.2, 88.5, 90.1, 89.4, 91.0, 92.2, 87.9, 90.5, 95.8, 89.9, 90.0, 50.0, 93.1, 91.5, 92.0])

print("--- Datos de Eficiencia del Sistema Robótico (15 Pruebas) ---")
print(eficiencia)
print("\n")

# 1. Media (Promedio) - numpy.mean()
# Es la suma de todos los valores dividida por el número de valores.
# Es muy sensible a valores atípicos (como el 50.0).
media = np.mean(eficiencia)
print(f"Media de Eficiencia: {media:.2f}%")

# 2. Mediana - numpy.median()
# Es el valor central cuando los datos están ordenados.
# Es robusta a valores atípicos (un mejor indicador del centro si hay fallos).
mediana = np.median(eficiencia)
print(f"Mediana de Eficiencia: {mediana:.2f}%")

# 3. Desviación Estándar - numpy.std()
# Mide la dispersión de los datos con respecto a la media.
# Una desviación estándar alta indica que los datos están muy dispersos, lo que puede significar inestabilidad.
desviacion_estandar = np.std(eficiencia)
print(f"Desviación Estándar (STD): {desviacion_estandar:.2f}")

# 4. Uso de Pandas para un resumen rápido (describe)
print("\n--- Resumen Estadístico Completo con Pandas ---")
df_eficiencia = pd.Series(eficiencia)
print(df_eficiencia.describe())