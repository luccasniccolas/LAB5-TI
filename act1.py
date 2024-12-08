import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# Parámetros de simulación
num_samples = int(1e6)  # Número de muestras a generar
K_values = [0, 10, 100]  # Factores K a considerar
sigma = 1  # Desviación estándar de la señal aleatoria (ruido)

# Crear la figura para las gráficas
plt.figure()

# Simulación para diferentes valores de K
for K in K_values:
    # Canal Riceano
    # La señal "determinística" es sqrt(K), y el ruido es una variable normal
    noise_real = sigma * np.random.randn(num_samples)  # Parte real del ruido
    noise_imag = sigma * np.random.randn(num_samples)  # Parte imaginaria del ruido
    noise_magnitude = np.sqrt(noise_real**2 + noise_imag**2)  # Magnitud del ruido

    # Componente Riceano
    channel_magnitude = np.sqrt(K) + noise_magnitude  # Señal total en magnitud

    # Calcular la CDF usando la función de distribución empírica
    ecdf = ECDF(channel_magnitude)  # CDF empírica
    x = ecdf.x  # Valores de la magnitud
    f = ecdf.y  # Valores de la CDF

    # Graficar la CDF
    plt.plot(x, f, label=f"K = {K}")

# Personalización de la gráfica
plt.xlabel('Magnitud de la señal')
plt.ylabel('CDF')
plt.legend()
plt.title('CDF del Canal Riceano para diferentes valores de K')
plt.grid()
plt.show()
