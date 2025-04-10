# Importación de todas las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

from ComparacionSignal import extraer_caracteristicas

# Configuración general de visualización
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Variables globales
FRECUENCIA_CORTE = 15  # Frecuencia de corte para el filtro pasa bajos
FRECUENCIA_MUESTREO = 173.61 # Frecuencia de muestreo en Hz

# =============================================
# 0. Carga de señales EEG
# =============================================

def cargar_senales():
    """Carga las tres señales de EEG desde archivos de texto"""
    senal_sana = np.loadtxt('Signal_1.txt')
    senal_interictal = np.loadtxt('Signal_2.txt')
    senal_convulsion = np.loadtxt('Signal_3.txt')
    return senal_sana, senal_interictal, senal_convulsion

# =============================================
# 1. Filtro de pasa bajos para eliminar el ruido. Se utiliza frecuencia de corte 20 Hz
# =============================================

def filtrar_senal(senal, frecuencia_corte= FRECUENCIA_CORTE, frecuencia_muestreo= FRECUENCIA_MUESTREO, orden_filtro=4):
    """Aplica filtro pasa bajos a la señal EEG"""
    nyquist = 0.5 * frecuencia_muestreo
    frecuencia_normalizada = frecuencia_corte / nyquist
    b, a = signal.butter(orden_filtro, frecuencia_normalizada, btype='low')
    return signal.filtfilt(b, a, senal)

# =============================================
# 2. Análisis en el dominio del tiempo
# =============================================

def graficar_senales_tiempo(senal_sana, senal_interictal, senal_convulsion, fs = FRECUENCIA_MUESTREO):
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

    Parámetros:
        frecuencias (array): Vector de frecuencias del espectro
        amplitud (array): Vector de amplitudes del espectro
    """
    # Calcular características espectrales
    caracteristicas = extraer_caracteristicas(frecuencias, amplitud)

    # Calcular porcentajes
    total = caracteristicas['potencia_total']
    if total == 0:
        print("La potencia total es cero - no se pueden calcular porcentajes")
        return

    # Diccionario de bandas con nombres formales
    bandas = {
        'delta': 'Delta (0.5-4 Hz)',
        'theta': 'Theta (4-8 Hz)',
        'alpha': 'Alpha (8-13 Hz)',
        'beta': 'Beta (13-30 Hz)',
        'gamma': 'Gamma (30-40 Hz)'
    }

    # Calcular y ordenar porcentajes
    porcentajes = []
    for banda in bandas:
        porcentaje = (caracteristicas[banda] / total) * 100
        porcentajes.append((bandas[banda], porcentaje))

    # Ordenar por porcentaje descendente
    porcentajes.sort(key=lambda x: x[1], reverse=True)

    # Mostrar resultados
    print("\nDistribución de potencia por bandas:")
    for nombre, porcentaje in porcentajes:
        print(f"- {nombre}: {porcentaje:.2f}%")

    # Mostrar ratios adicionales
    print("\nRatios importantes:")
    print(f"Alpha/Theta: {caracteristicas['ratio_alpha_theta']:.2f}")
    print(f"Beta/Alpha: {caracteristicas['ratio_beta_alpha']:.2f}")
    print(f"Gamma/Alpha: {caracteristicas['ratio_gamma_alpha']:.2f}")

# =============================================
# 3. Análisis en el dominio de la frecuencia
# =============================================

def calcular_espectro_frecuencias(senal, fs=173.61):
    """Calcula la transformada de Fourier de la señal"""
    n = len(senal)
    yf = fft(senal)
    xf = fftfreq(n, 1/fs)[:n//2]
    return xf, 2/n * np.abs(yf[0:n//2])

def graficar_espectro_frecuencias(xf, yf, titulo, limite_superior=50):
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

    return {nombre: np.trapz(yf[(xf >= fmin) & (xf <= fmax)],
                            xf[(xf >= fmin) & (xf <= fmax)])
            for nombre, (fmin, fmax) in bandas.items()}

def graficar_comparacion_potencias(potencias_sana, potencias_interictal, potencias_convulsion):
    """Comparación visual de potencias por bandas"""
    nombres_bandas = list(potencias_sana.keys())
    x = np.arange(len(nombres_bandas))
    ancho = 0.25

    plt.figure(figsize=(12,6))
    plt.bar(x - ancho, potencias_sana.values(), ancho, label='Sana')
    plt.bar(x, potencias_interictal.values(), ancho, label='Interictal')
    plt.bar(x + ancho, potencias_convulsion.values(), ancho, label='Convulsión')

    plt.xticks(x, nombres_bandas, rotation=45)
    plt.ylabel('Potencia espectral')
    plt.title('Distribución de potencia por bandas de frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================
# 5. Análisis de autocorrelación
# =============================================

def calcular_autocorrelacion(senal, max_retardo=1000):
    """Calcula la autocorrelación normalizada"""
    autocorr = np.correlate(senal, senal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    return autocorr[:max_retardo] / autocorr[0]

def graficar_autocorrelaciones(autocorr_sana, autocorr_interictal, autocorr_convulsion):
    """Visualización comparativa de autocorrelaciones"""
    retardos = np.arange(len(autocorr_sana))

    plt.figure(figsize=(12,6))
    plt.plot(retardos, autocorr_sana, label='Sana')
    plt.plot(retardos, autocorr_interictal, label='Interictal')
    plt.plot(retardos, autocorr_convulsion, label='Convulsión')

    plt.title('Comparación de Autocorrelaciones')
    plt.xlabel('Retardo (muestras)')
    plt.ylabel('Autocorrelación Normalizada')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================
# Función principal de análisis
# =============================================

def graficar_comparacion_tiempo(senal_original, senal_filtrada, titulo, fs=173.61):
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


def analisis_completo_eeg():
    # 1. Carga y filtrado de señales
    
    # CARGA DE SEÑALES
    senal_sana, senal_interictal, senal_convulsion = cargar_senales()

    # FILTRO PASA BAJOS CON CORTE = 15 HZ
    senal_sana_f = filtrar_senal(senal_sana)
    senal_interictal_f = filtrar_senal(senal_interictal)
    senal_convulsion_f = filtrar_senal(senal_convulsion)

    #GRAFICA DE SEÑALES ORIGINALES Y FILTRADAS
    print("\nComparación señales originales vs filtradas...")
    graficar_comparacion_tiempo(senal_sana, senal_sana_f, 'Señal Sana')
    graficar_comparacion_tiempo(senal_interictal, senal_interictal_f, 'Señal Interictal')
    graficar_comparacion_tiempo(senal_convulsion, senal_convulsion_f, 'Señal de Convulsión')
    
    # 2. Aplicar la transformada de Fourier a cada una de las senales
    print("\nAnalizando distribución espectral...")
    espectros = {
         'Sana': (senal_sana_f, 'Señal Sana'),
         'Interictal': (senal_interictal_f, 'Señal Interictal'),
         'Convulsion': (senal_convulsion_f, 'Señal Convulsión')
     }
    
    for clave, (senal, nombre) in espectros.items():
         xf, yf = calcular_espectro_frecuencias(senal)
         print(f"\n==== Análisis detallado - {nombre} ====")
         analizar_distribucion_bandas(xf, yf)
         graficar_espectro_frecuencias(xf, yf, nombre)


# # 4. Potencia por bandas
    # print("\nCalculando potencia por bandas espectrales...")
    # potencias_sana = calcular_potencia_bandas(xf_sana, yf_sana)
    # potencias_inter = calcular_potencia_bandas(xf_inter, yf_inter)
    # potencias_conv = calcular_potencia_bandas(xf_conv, yf_conv)
    #
    # graficar_comparacion_potencias(potencias_sana, potencias_inter, potencias_conv)
    #
    # # 5. Autocorrelación
    # print("\nCalculando autocorrelaciones...")
    # autocorr_sana = calcular_autocorrelacion(senal_sana_f)
    # autocorr_inter = calcular_autocorrelacion(senal_interictal_f)
    # autocorr_conv = calcular_autocorrelacion(senal_convulsion_f)
    #
    # graficar_autocorrelaciones(autocorr_sana, autocorr_inter, autocorr_conv)
    #
    # print("\nAnálisis completado.")

# Ejecución del análisis completo
if __name__ == "__main__":
    analisis_completo_eeg()