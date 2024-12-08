import numpy as np
import matplotlib.pyplot as plt

# Umbrales de potencia en dB
thresholds_db = [3, 5, 10, 20]
thresholds_linear = [10**(db/10) for db in thresholds_db]

# Parámetros del canal
K_values = [0, 10, 100]  # Factores K a considerar
sigma = 1  # Desviación estándar del ruido
num_samples = int(1e6)  # Número de muestras

# Inicializar resultados
lcr_results = {K: [] for K in K_values}
afd_results = {K: [] for K in K_values}

# Simulación para diferentes valores de K
for K in K_values:
    # Generar ruido gaussiano
    noise_real = sigma * np.random.randn(num_samples)
    noise_imag = sigma * np.random.randn(num_samples)
    noise_magnitude = np.sqrt(noise_real**2 + noise_imag**2)

    # Componente Riceano
    channel_magnitude = np.sqrt(K) + noise_magnitude

    # Calcular LCR y AFD para cada umbral
    for threshold in thresholds_linear:
        # Calcular LCR
        upward_crossings = np.sum((channel_magnitude[:-1] < threshold) & (channel_magnitude[1:] >= threshold))
        lcr = upward_crossings / (num_samples - 1)
        lcr_results[K].append(lcr)

        # Calcular AFD
        below_threshold = channel_magnitude < threshold
        fade_durations = np.diff(np.where(np.concatenate(([0], below_threshold, [0])))[0]) - 1
        avg_fade_duration = np.mean(fade_durations[fade_durations > 0]) if np.any(fade_durations > 0) else 0
        afd_results[K].append(avg_fade_duration)

# Graficar resultados de LCR
plt.figure()
for K, lcr_values in lcr_results.items():
    plt.plot(thresholds_db, lcr_values, label=f"K = {K}")
plt.xlabel('Umbral de Potencia (dB)')
plt.ylabel('Tasa de Cruces (LCR)')
plt.title('LCR vs Factor K para diferentes umbrales de potencia')
plt.legend()
plt.grid()
plt.show()

# Graficar resultados de AFD
plt.figure()
for K, afd_values in afd_results.items():
    plt.plot(thresholds_db, afd_values, label=f"K = {K}")
plt.xlabel('Umbral de Potencia (dB)')
plt.ylabel('Duración Promedio de Desvanecimientos (AFD)')
plt.title('AFD vs Factor K para diferentes umbrales de potencia')
plt.legend()
plt.grid()
plt.show()
