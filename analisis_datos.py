import pandas as pd
import numpy as np

# 1. Crear datos simulados para Mecatrónica
data = {
    'Tiempo_Seg': np.arange(1, 11),  # De 1 a 10 segundos
    'Temperatura_C': [80, 82, 85, 87, 86, 90, 93, 91, 95, 98], # Datos de un sensor de temperatura
    'Vibracion_mm_s': [0.5, 0.6, 0.55, 0.7, 1.2, 1.8, 2.5, 2.6, 3.1, 4.0] # Datos de un acelerómetro
}

# 2. Crear un DataFrame de Pandas (Estructura tabular)
df = pd.DataFrame(data)
print("--- DataFrame de Sensores ---")
print(df)
print("\n")

# 3. Aplicar una Operación de NumPy
# Usaremos NumPy para calcular la Raíz Cuadrada (un tipo de normalización)
# de los valores de vibración, un paso común antes de entrenar modelos de IA.
vibracion_array = df['Vibracion_mm_s'].values # Convertir la columna a un array de NumPy
raiz_vibracion = np.sqrt(vibracion_array)

# 4. Agregar el resultado al DataFrame
df['Raiz_Vibracion'] = raiz_vibracion

print("--- Análisis NumPy (Raíz Cuadrada de Vibración) ---")
print(df[['Vibracion_mm_s', 'Raiz_Vibracion']])

# 5. Calcular la media con NumPy para toda la columna de temperatura
media_temperatura = np.mean(df['Temperatura_C'].values)
print(f"\nMedia de Temperatura: {media_temperatura:.2f} C")