# funciones_eeg.py

# Importación de bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from ComparacionSignal import extraer_caracteristicas

from env import (
    FRECUENCIA_CORTE,
    FRECUENCIA_MUESTREO,
    BANDAS_EEG,
    LIMITE_FRECUENCIAS,
    MAX_RETARDO
)
# =============================================
# 0. Carga de señales EEG
# =============================================

def cargar_senales():
    senal_sana = np.loadtxt('Signal_1.txt')
    senal_interictal = np.loadtxt('Signal_2.txt')
    senal_convulsion = np.loadtxt('Signal_3.txt')
    return senal_sana, senal_interictal, senal_convulsion

# =============================================
# 1. Filtro de pasa bajos para eliminar el ruido
# =============================================

def filtrar_senal(senal, frecuencia_corte=FRECUENCIA_CORTE, frecuencia_muestreo=FRECUENCIA_MUESTREO, orden_filtro=4):
    nyquist = 0.5 * frecuencia_muestreo
    frecuencia_normalizada = frecuencia_corte / nyquist
    b, a = signal.butter(orden_filtro, frecuencia_normalizada, btype='low')
    return signal.filtfilt(b, a, senal)

# =============================================
# 2. Análisis en el dominio del tiempo
# =============================================
def graficar_senal_original_y_filtrada_con_transformada(senal_original, senal_filtrada, titulo):
    """Grafica la señal original y filtrada juntas en el dominio del tiempo y la transformada de Fourier de la señal original"""
    # Calcular transformada de Fourier de la señal original
    fs = FRECUENCIA_MUESTREO
    xf, yf = calcular_espectro_frecuencias(senal_filtrada, fs)

    # Crear subgráficos
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Señal original y filtrada en el dominio del tiempo
    tiempo = np.arange(len(senal_original)) / fs
    axs[0].plot(tiempo, senal_original, color='blue', label='Original', alpha=0.8)
    axs[0].plot(tiempo, senal_filtrada, color='orange', label='Filtrada', alpha=0.8)
    axs[0].set_title(f'{titulo} - Dominio del Tiempo')
    axs[0].set_xlabel('Tiempo (s)')
    axs[0].set_ylabel('Amplitud (μV)')
    axs[0].legend()
    axs[0].grid(True)

    # Transformada de Fourier de la señal original
    axs[1].plot(xf, yf, color='blue')
    axs[1].set_title(f'{titulo} - Espectro de Frecuencia (Señal Original)')
    axs[1].set_xlabel('Frecuencia (Hz)')
    axs[1].set_ylabel('Amplitud Normalizada')
    axs[1].set_xlim(0, 50)  # Limitar a 50 Hz para señales EEG
    axs[1].grid(True)

    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def calcular_espectro_frecuencias(senal, fs=FRECUENCIA_MUESTREO):
    """Calcula la transformada de Fourier de la señal"""
    n = len(senal)
    yf = fft(senal)
    xf = fftfreq(n, 1/fs)[:n//2]
    return xf, 2/n * np.abs(yf[0:n//2])

def graficar_espectro_frecuencias(xf, yf, titulo, limite_superior=LIMITE_FRECUENCIAS):
    """Visualización del espectro de frecuencias"""
    plt.figure()
    plt.plot(xf, yf)
    plt.title(f'Espectro de Frecuencia - {titulo}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud Normalizada')
    plt.xlim(0, limite_superior)
    plt.grid(True)
    plt.show()

# =============================================
# 4. Análisis de potencia espectral por bandas
# =============================================

def calcular_potencia_bandas(xf, yf):
    """Calcula la potencia relativa en las bandas típicas de EEG"""
    bandas = BANDAS_EEG
    potencias = {}
    total = np.trapezoid(yf, xf)  # Potencia total
    for nombre, (fmin, fmax) in bandas.items():
        mascara = (xf >= fmin) & (xf <= fmax)
        potencia_abs = np.trapezoid(yf[mascara], xf[mascara])
        potencias[nombre] = potencia_abs / total  # Normalización
    return potencias

def graficar_comparacion_potencias(potencias_sana, potencias_interictal, potencias_convulsion):
    """Comparación visual de potencias por bandas con promedio por estado"""
    nombres_bandas = list(potencias_sana.keys())
    x = np.arange(len(nombres_bandas))
    ancho = 0.25

    # Calcular promedios
    promedio_sana = sum(potencias_sana.values()) / len(potencias_sana)
    promedio_interictal = sum(potencias_interictal.values()) / len(potencias_interictal)
    promedio_convulsion = sum(potencias_convulsion.values()) / len(potencias_convulsion)

    # Imprimir valores de potencia y promedios
    print("\nPotencias por bandas:")
    print("Sana:")
    for banda, potencia in potencias_sana.items():
        print(f"  {banda}: {potencia:.4f}")
    print(f"  **Promedio total**: {promedio_sana:.4f}")  # <-- Nuevo

    print("\nInterictal:")
    for banda, potencia in potencias_interictal.items():
        print(f"  {banda}: {potencia:.4f}")
    print(f"  **Promedio total**: {promedio_interictal:.4f}")  # <-- Nuevo

    print("\nConvulsión:")
    for banda, potencia in potencias_convulsion.items():
        print(f"  {banda}: {potencia:.4f}")
    print(f"  **Promedio total**: {promedio_convulsion:.4f}")  # <-- Nuevo

    # Graficar potencias (el resto del código se mantiene igual)
    plt.figure(figsize=(12, 6))
    barras_sana = plt.bar(x - ancho, potencias_sana.values(), ancho, label='Sana')
    barras_inter = plt.bar(x, potencias_interictal.values(), ancho, label='Interictal')
    barras_conv = plt.bar(x + ancho, potencias_convulsion.values(), ancho, label='Convulsión')

    for barras in [barras_sana, barras_inter, barras_conv]:
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, altura, f'{altura:.2f}',
                     ha='center', va='bottom', fontsize=10)

    plt.xticks(x, nombres_bandas, rotation=45)
    plt.ylabel('Potencia relativa (%)')
    plt.title('Distribución de potencia por bandas de frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()

def calcular_autocorrelacion(senal):
    n = len(senal)
    senal_centrada = senal - np.mean(senal)  
    #Centrar la señal
    #La función np.mean de NumPy calcula la media aritmética de los valores en el array senal.
    #La media es el promedio de todos los valores en la señal, lo que representa el "nivel base" o el valor promedio de la señal.
    autocorr = np.correlate(senal_centrada, senal_centrada, mode='full')
    autocorr_normalizada = autocorr / autocorr[n - 1]  # Normalizar al valor en retardo 0
    lags = np.arange(-n + 1, n)  # Retardos desde -(n-1) hasta +(n-1) en muestras
    return lags, autocorr_normalizada
    
def graficar_autocorrelacion_con_senal_original(senal, autocorrelacion, lags, titulo):
    """Grafica la señal original junto con su autocorrelación en unidades de muestras"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Señal original
    axs[0].plot(senal, color='blue', label='Señal Original')
    axs[0].set_title(f'{titulo} - Señal Original')
    axs[0].set_xlabel('Muestras')
    axs[0].set_ylabel('Amplitud (μV)')
    axs[0].legend()
    axs[0].grid(True)

    # Autocorrelación (centrada en 0)
    axs[1].plot(lags, autocorrelacion, color='orange', label='Autocorrelación')
    axs[1].set_title(f'{titulo} - Autocorrelación')
    axs[1].set_xlabel('Retardo (muestras)')
    axs[1].set_ylabel('Autocorrelación Normalizada')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(-4500, 4500)  # Ajusta el rango según tus datos
    
    plt.tight_layout()
    plt.show()
