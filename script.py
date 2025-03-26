import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

# Configuración general
FS = 173.61  # Frecuencia de muestreo (Hz)
CUTOFF = 40   # Frecuencia de corte del filtro (Hz)
ORDER = 4     # Orden del filtro

# Bandas de frecuencia relevantes
BANDAS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

def cargar_senal(archivo):
    with open(archivo, 'r') as f:
        datos = [float(linea.strip()) for linea in f if linea.strip()]
    return np.array(datos)

def aplicar_filtro(senal, fs, cutoff, order):
    b, a = signal.butter(order, cutoff/(fs/2), btype='low')
    senal_filtrada = signal.filtfilt(b, a, senal)
    return senal_filtrada

def calcular_fft(senal, fs):
    N = len(senal)
    yf = rfft(senal)
    xf = rfftfreq(N, 1/fs)
    return xf, np.abs(yf)

def potencia_espectral(xf, yf, bandas):
    potencia = {}
    for nombre, (f_low, f_high) in bandas.items():
        mask = (xf >= f_low) & (xf <= f_high)
        potencia[nombre] = np.sum(yf[mask]**2)
    return potencia

def autocorrelacion(senal):
    corr = np.correlate(senal, senal, mode='full')
    return corr[len(corr)//2:]

def graficar_analisis(senal, senal_filtrada, xf, yf, potencia, autocorr, nombre):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Análisis de señal: {nombre}')
    
    # Señal original vs filtrada
    axs[0].plot(senal, label='Original', alpha=0.5)
    axs[0].plot(senal_filtrada, label='Filtrada', color='red')
    axs[0].set_title('Señal temporal')
    axs[0].legend()
    
    # Espectro de frecuencia
    axs[1].plot(xf, yf)
    axs[1].set_title('Espectro de frecuencia (FFT)')
    axs[1].set_xlabel('Frecuencia (Hz)')
    
    # Potencia espectral
    axs[2].bar(potencia.keys(), potencia.values())
    axs[2].set_title('Potencia espectral por bandas')
    
    # Autocorrelación
    axs[3].plot(autocorr)
    axs[3].set_title('Autocorrelación')
    
    plt.tight_layout()
    plt.savefig(f'analisis_{nombre}.png')
    plt.close()

def analizar_senal(archivo):
    # Cargar y procesar datos
    nombre = archivo.split('.')[0]
    senal = cargar_senal(archivo)
    senal = signal.detrend(senal)  # Remover tendencia lineal
    
    # Aplicar filtro
    senal_filtrada = aplicar_filtro(senal, FS, CUTOFF, ORDER)
    
    # Análisis FFT
    xf, yf = calcular_fft(senal_filtrada, FS)
    
    # Potencia espectral
    potencia = potencia_espectral(xf, yf, BANDAS)
    
    # Autocorrelación
    autocorr = autocorrelacion(senal_filtrada)
    autocorr = autocorr / np.max(autocorr)  # Normalizar
    
    # Generar gráficos
    graficar_analisis(senal, senal_filtrada, xf, yf, potencia, autocorr, nombre)
    
    return {
        'potencia': potencia,
        'autocorrelacion': autocorr
    }

# Análisis de todas las señales
if __name__ == '__main__':
    resultados = {}
    for archivo in ['Signal_1.txt', 'Signal_2.txt', 'Signal_3.txt']:
        resultados[archivo] = analizar_senal(archivo)
    
    # Comparación entre señales
    fig, ax = plt.subplots(figsize=(10, 5))
    for archivo, datos in resultados.items():
        ax.plot(datos['autocorrelacion'][:200], label=archivo)
    ax.set_title('Comparación de autocorrelaciones')
    ax.legend()
    plt.savefig('comparacion_autocorrelaciones.png')
    plt.close()