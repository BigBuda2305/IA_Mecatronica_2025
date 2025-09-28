import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn es una capa de Matplotlib para gráficos más atractivos

# 1. Recrear el DataFrame del Día 3
data = {
    'Tiempo_Seg': np.arange(1, 11),
    'Temperatura_C': [80, 82, 85, 87, 86, 90, 93, 91, 95, 98],
    'Vibracion_mm_s': [0.5, 0.6, 0.55, 0.7, 1.2, 1.8, 2.5, 2.6, 3.1, 4.0]
}
df = pd.DataFrame(data)

# 2. Configuración de Seaborn
sns.set_style("whitegrid") # Estilo estético para los gráficos

# --- GRÁFICO 1: Diagrama de Dispersión (Scatter Plot) ---
# Muestra si existe una correlación entre Temperatura y Vibración
plt.figure(figsize=(8, 6)) # Define el tamaño de la figura
sns.scatterplot(x='Temperatura_C', y='Vibracion_mm_s', data=df, s=100, color='red')
plt.title('Vibración vs. Temperatura (Correlación en Motor)')
plt.xlabel('Temperatura del Motor (°C)')
plt.ylabel('Vibración (mm/s)')
plt.grid(True)

# 3. Mostrar la primera figura antes de crear la segunda
# plt.show() # Si lo descomentas, la figura aparecerá, pero la detendremos para juntar los gráficos

# --- GRÁFICO 2: Gráfico de Líneas (Time Series) ---
# Muestra la tendencia de la Vibración a lo largo del tiempo
plt.figure(figsize=(10, 4))
sns.lineplot(x='Tiempo_Seg', y='Vibracion_mm_s', data=df, marker='o')
plt.title('Tendencia de la Vibración del Robot a lo largo del Tiempo')
plt.xlabel('Tiempo (Segundos)')
plt.ylabel('Vibración (mm/s)')

# 4. Mostrar todas las figuras
plt.tight_layout() # Ajusta los gráficos para que no se superpongan
plt.show()