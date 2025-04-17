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

def graficar_senales_tiempo(senal_sana, senal_interictal, senal_convulsion, fs=FRECUENCIA_MUESTREO):
    """Visualización comparativa de las señales en tiempo"""
    tiempo = lambda s: np.arange(len(s)) / fs

    fig, axs = plt.subplots(3, 1, figsize=(15,10))

    axs[0].plot(tiempo(senal_sana), senal_sana)
    axs[0].set_title('Señal Sana (Estado Normal)')
    axs[0].set_ylabel('Amplitud (μV)')

    axs[1].plot(tiempo(senal_interictal), senal_interictal)
    axs[1].set_title('Señal Interictal (Entre Crisis)')
    axs[1].set_ylabel('Amplitud (μV)')

    axs[2].plot(tiempo(senal_convulsion), senal_convulsion)
    axs[2].set_title('Señal de Convulsión (Crisis Epiléptica)')
    axs[2].set_xlabel('Tiempo (s)')
    axs[2].set_ylabel('Amplitud (μV)')

    plt.tight_layout()
    plt.show()

def analizar_distribucion_bandas(frecuencias, amplitud):
    """
    Muestra el porcentaje de contribución de todas las bandas espectrales
    """
    caracteristicas = extraer_caracteristicas(frecuencias, amplitud)

    total = caracteristicas['potencia_total']
    if total == 0:
        print("La potencia total es cero - no se pueden calcular porcentajes")
        return

    bandas = {
        'delta': 'Delta (0.5-4 Hz)',
        'theta': 'Theta (4-8 Hz)',
        'alpha': 'Alpha (8-13 Hz)',
        'beta': 'Beta (13-30 Hz)',
        'gamma': 'Gamma (30-40 Hz)'
    }

    porcentajes = []
    for banda in bandas:
        porcentaje = (caracteristicas[banda] / total) * 100
        porcentajes.append((bandas[banda], porcentaje))

    porcentajes.sort(key=lambda x: x[1], reverse=True)

    print("\nDistribución de potencia por bandas:")
    for nombre, porcentaje in porcentajes:
        print(f"- {nombre}: {porcentaje:.2f}%")

    print("\nRatios importantes:")
    print(f"Alpha/Theta: {caracteristicas['ratio_alpha_theta']:.2f}")
    print(f"Beta/Alpha: {caracteristicas['ratio_beta_alpha']:.2f}")
    print(f"Gamma/Alpha: {caracteristicas['ratio_gamma_alpha']:.2f}")


# =============================================
# . Análisis en el dominio de la frecuencia
# =============================================

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
    """Calcula la potencia en las bandas típicas de EEG"""
    bandas = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-40 Hz)': (30, 40)
    }
    #
    # return {nombre: np.trapezoid(yf[(xf >= fmin) & (xf <= fmax)], xf[(xf >= fmin) & (xf <= fmax)])
    #         for nombre, (fmin, fmax) in bandas.items()}
    potencias = {}
    total = np.trapezoid(yf, xf)  # Potencia total

    for nombre, (fmin, fmax) in bandas.items():
        potencia = np.trapezoid(yf[(xf >= fmin) & (xf <= fmax)], xf[(xf >= fmin) & (xf <= fmax)])
        potencias[nombre] = (potencia / total)
    return potencias

def graficar_comparacion_potencias(potencias_sana, potencias_interictal, potencias_convulsion):
    """Comparación visual de potencias por bandas"""
    nombres_bandas = list(potencias_sana.keys())
    x = np.arange(len(nombres_bandas))
    ancho = 0.25

    # Imprimir valores de potencia
    print("\nPotencias por bandas:")
    print("Sana:")
    for banda, potencia in potencias_sana.items():
        print(f"  {banda}: {potencia:.4f}")
    print("Interictal:")
    for banda, potencia in potencias_interictal.items():
        print(f"  {banda}: {potencia:.4f}")
    print("Convulsión:")
    for banda, potencia in potencias_convulsion.items():
        print(f"  {banda}: {potencia:.4f}")

    # Graficar potencias
    plt.figure(figsize=(12, 6))
    barras_sana = plt.bar(x - ancho, potencias_sana.values(), ancho, label='Sana')
    barras_inter = plt.bar(x, potencias_interictal.values(), ancho, label='Interictal')
    barras_conv = plt.bar(x + ancho, potencias_convulsion.values(), ancho, label='Convulsión')

    # Agregar valores encima de las barras
    for barras in [barras_sana, barras_inter, barras_conv]:
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, altura, f'{altura:.2f}',
                     ha='center', va='bottom', fontsize=10)

    plt.xticks(x, nombres_bandas, rotation=45)
    plt.ylabel('Potencia espectral (uV)')
    plt.title('Distribución de potencia por bandas de frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()
# =============================================
# 5. Análisis de autocorrelación
# =============================================

def calcular_autocorrelacion(senal):
    n = len(senal)
    senal_centrada = senal - np.mean(senal)  # Centrar la señal
    #La función np.mean de NumPy calcula la media aritmética de los valores en el array senal.
    #La media es el promedio de todos los valores en la señal, lo que representa el "nivel base" o el valor promedio de la señal.
    autocorr = np.correlate(senal_centrada, senal_centrada, mode='full')
    autocorr_normalizada = autocorr / autocorr[n - 1]  # Normalizar al valor en retardo 0
    lags = np.arange(-n + 1, n)  # Retardos desde -(n-1) hasta +(n-1) en muestras
    return lags, autocorr_normalizada

def graficar_autocorrelaciones(autocorr_sana, autocorr_interictal, autocorr_convulsion):
    """Visualización comparativa de autocorrelaciones"""
    retardosSana = np.arange(len(autocorr_sana))
    retardosInt = np.arange(len(autocorr_interictal))
    retardosConv = np.arange(len(autocorr_convulsion))

    plt.figure(figsize=(12,6))
    plt.plot(retardosSana, autocorr_sana, label='Sana')
    plt.plot(retardosInt, autocorr_interictal, label='Interictal')
    plt.plot(retardosConv, autocorr_convulsion, label='Convulsión')

    plt.title('Comparación de Autocorrelaciones')
    plt.xlabel('Retardo (muestras)')
    plt.ylabel('Autocorrelación Normalizada')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
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

# =============================================
# Función auxiliar para comparación en tiempo
# =============================================

def graficar_comparacion_tiempo(senal_original, senal_filtrada, titulo, fs=FRECUENCIA_MUESTREO):
    """Grafica señal original y filtrada superpuestas"""
    tiempo = np.arange(len(senal_original)) / fs

    plt.figure(figsize=(15, 5))
    plt.plot(tiempo, senal_original, alpha=0.8, label='Original', linewidth=1.5, color='darkblue')
    plt.plot(tiempo, senal_filtrada, label='Filtrada', linewidth=1.2, color='orange')

    plt.title(f'Comparación: {titulo}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (μV)')
    plt.legend()
    plt.grid(True)
    plt.show()