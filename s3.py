import numpy as np
import matplotlib.pyplot as plt

def cargar_senal(archivo):
    with open(archivo, 'r') as f:
        return np.array([float(linea.strip()) for linea in f])

senal = cargar_senal('Signal_3.txt')
tiempo = np.arange(len(senal)) / 173.61  # Vector de tiempo en segundos

plt.figure(figsize=(12, 6))
plt.plot(tiempo, senal, 'b', linewidth=0.5)
plt.title('EEG durante crisis epiléptica (Signal_3.txt)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (µV)')
plt.grid(alpha=0.3)
plt.show()