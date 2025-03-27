from matplotlib.pylab import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

# Configuración
FS = 173.61
CUTOFF = 40
ORDER = 4

# Cargar señal
def cargar_senal(archivo):
    with open(archivo, 'r') as f:
        return np.array([float(linea.strip()) for linea in f])

senal_original = cargar_senal('Signal_1.txt')
senal_original = signal.detrend(senal_original)

# Crear y aplicar filtro
b, a = butter(ORDER, CUTOFF/(FS/2), btype='low')
senal_filtrada = filtfilt(b, a, senal_original)

# Calcular diferencia
diferencia = senal_original - senal_filtrada

# Gráficos
plt.figure(figsize=(15, 8))

# Señal original vs filtrada
plt.subplot(3, 1, 1)
plt.plot(senal_original, 'b', alpha=0.5, label='Original')
plt.plot(senal_filtrada, 'r', linewidth=1.5, label='Filtrada')
plt.title('Comparación temporal')
plt.legend()

# Diferencia
plt.subplot(3, 1, 2)
plt.plot(diferencia, 'g', label='Diferencia (Original - Filtrada)')
plt.title('Componentes eliminados por el filtro')
plt.legend()

# Espectro de diferencia
plt.subplot(3, 1, 3)
freqs = rfftfreq(len(diferencia), 1/FS)
plt.plot(freqs, np.abs(rfft(diferencia)), 'm')
plt.title('Espectro de frecuencias de la diferencia')
plt.xlabel('Frecuencia (Hz)')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# Error cuadrático medio (RMSE)
rmse = np.sqrt(np.mean(diferencia**2))
print(f'RMSE entre original y filtrada: {rmse:.2f} µV')

# Correlación
correlacion = np.corrcoef(senal_original, senal_filtrada)[0, 1]
print(f'Coeficiente de correlación: {correlacion:.4f}')