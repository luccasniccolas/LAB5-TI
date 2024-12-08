import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación
K_values = [0, 10, 100]  # Factores K a evaluar
thresholds_dB = [3, 5, 10, 20]  # Umbrales de potencia en dB
time_duration = 10  # Duración de la simulación en segundos
fs = 1e3  # Frecuencia de muestreo (Hz)
t = np.arange(0, time_duration, 1/fs)  # Vector de tiempo

# Matriz para almacenar resultados de LCR
LCR_results = np.zeros((len(K_values), len(thresholds_dB)))

# Bucle para calcular LCR para cada valor de K y cada umbral
for i, K in enumerate(K_values):
    # Generar el canal Riceano para el valor de K
    s = np.sqrt(K / (K + 1))  # Componente de línea de visión
    sigma = np.sqrt(1 / (2 * (K + 1)))  # Desviación estándar de los componentes reflejados

    # Generar señal Riceana
    noise_real = sigma * np.random.randn(len(t))  # Parte real
    noise_imag = sigma * np.random.randn(len(t))  # Parte imaginaria
    rice_signal = np.sqrt((s + noise_real)**2 + noise_imag**2)  # Magnitud de la señal

    # Convertir la magnitud de la señal a dB
    rice_signal_dB = 20 * np.log10(rice_signal)

    # Calcular LCR para cada umbral
    for j, threshold in enumerate(thresholds_dB):
        # Contar cruces descendentes por el umbral
        is_above = rice_signal_dB > threshold
        crossings = np.where(np.diff(is_above.astype(int)) == -1)[0]
        LCR_results[i, j] = len(crossings) / time_duration  # Cruces por segundo

# Graficar LCR vs. Factor K para cada umbral
plt.figure(figsize=(8, 6))
for j, threshold in enumerate(thresholds_dB):
    plt.plot(K_values, LCR_results[:, j], '-o', label=f'Umbral {threshold} dB')

plt.xlabel('Factor K')
plt.ylabel('LCR (Cruces por segundo)')
plt.title('LCR vs. Factor K para diferentes umbrales de potencia')
plt.legend()
plt.grid(True)
plt.show()
