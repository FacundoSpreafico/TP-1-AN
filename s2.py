import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def cargar_txt(ruta_archivo):
    """
    Carga una señal de un archivo de texto (.txt) y devuelve un array de NumPy.
    
    Parámetros:
        ruta_archivo (str): La ruta al archivo .txt que contiene la señal.
    
    Retorna:
        np.ndarray: Array con los datos de la señal.
    """
    try:
        # Se utiliza numpy.loadtxt para leer el archivo
        señal = np.loadtxt(ruta_archivo)
        print(f"Señal cargada exitosamente desde {ruta_archivo}.")
        return señal
    except Exception as e:
        print(f"Error al cargar el archivo {ruta_archivo}: {e}")
        return None


senial1 = cargar_txt('Signal_1.txt')
senial2 = cargar_txt('Signal_2.txt')

# Parámetros
fs = 173.61  # Frecuencia de muestreo en Hz
cutoff = 40  # Frecuencia de corte en Hz

# Diseño del filtro (orden 4, por ejemplo)
order = 4
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Suponiendo que 'signal' es un array con la señal EEG
signal_filtrada = filtfilt(b, a, senial1)

# Ejemplo de visualización
plt.figure(figsize=(10, 4))
plt.plot(signal, label='Señal original')
plt.plot(signal_filtrada, label='Señal filtrada', linewidth=2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Filtrado pasa bajos a 40 Hz')
plt.legend()
plt.show()
