import numpy as np
import matplotlib.pyplot as plt

# Función para calcular el Average Fade Duration (AFD)
def calculate_afd(signal_dB, threshold):
    below_threshold = signal_dB < threshold
    fade_durations = []
    duration = 0

    # Calculamos las duraciones de los desvanecimientos
    for i in range(1, len(below_threshold)):
        if below_threshold[i] and not below_threshold[i-1]:
            # Inicio de un fade
            duration = 1
        elif below_threshold[i] and below_threshold[i-1]:
            # Continúa el fade
            duration += 1
        elif not below_threshold[i] and below_threshold[i-1]:
            # Fin del fade
            fade_durations.append(duration)
            duration = 0

    # Añadir la última duración si el fade terminó al final
    if duration > 0:
        fade_durations.append(duration)
    
    # Calcular el AFD como la duración promedio de los fades
    return np.mean(fade_durations) if fade_durations else 0

# Parámetros de simulación
K_values = [0, 10, 100]  # Factores K a evaluar
thresholds_dB = [3, 5, 10, 20]  # Umbrales de potencia en dB
time_duration = 10  # Duración de la simulación en segundos
fs = 1e3  # Frecuencia de muestreo (Hz)
t = np.arange(0, time_duration, 1/fs)  # Vector de tiempo

# Matriz para almacenar resultados de AFD
AFD_results = np.zeros((len(K_values), len(thresholds_dB)))

# Bucle para calcular AFD para cada valor de K y cada umbral
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

    # Calcular AFD para cada umbral
    for j, threshold in enumerate(thresholds_dB):
        afd = calculate_afd(rice_signal_dB, threshold)
        AFD_results[i, j] = afd


print("Valores de AFD para cada umbral y K:")
print(AFD_results)

# Graficar AFD vs. Factor K para cada umbral
plt.figure(figsize=(8, 6))
for j, threshold in enumerate(thresholds_dB):
    plt.plot(K_values, AFD_results[:, j], '-o', label=f'Umbral {threshold} dB')

plt.xlabel('Factor K')
plt.ylabel('AFD (Duración promedio del desvanecimiento)')
plt.title('AFD vs. Factor K para diferentes umbrales de potencia')
plt.legend()
plt.grid(True)
plt.show()
