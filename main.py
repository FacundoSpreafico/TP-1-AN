# main.py

# Importación de bibliotecas
from matplotlib.mlab import psd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from env import FRECUENCIA_MUESTREO, VISUAL_CONFIG

# Aplicar configuración visual
plt.rcParams.update(VISUAL_CONFIG)

from funciones_eeg import (
    cargar_senales, filtrar_senal, graficar_autocorrelacion_con_senal_original,
    calcular_espectro_frecuencias, calcular_potencia_bandas,
    graficar_comparacion_potencias, calcular_autocorrelacion,
    graficar_senal_original_y_filtrada_con_transformada,
    calcular_potencia_espectral
)
# =============================================
# Función principal de análisis
# =============================================

def analisis_completo_eeg():
    # 1. Carga
    senal_sana, senal_interictal, senal_convulsion = cargar_senales()
    
    # 2. Filtrado de señales
    senal_sana_f = filtrar_senal(senal_sana)
    senal_interictal_f = filtrar_senal(senal_interictal)
    senal_convulsion_f = filtrar_senal(senal_convulsion)

    # Graficar comparación original vs filtrado
    print("\nComparación señales originales vs filtradas...")
    graficar_senal_original_y_filtrada_con_transformada(senal_sana, senal_sana_f , 'Senal Sana filtrada')
    graficar_senal_original_y_filtrada_con_transformada(senal_interictal, senal_interictal_f, 'Senal Interictal filtrada')
    graficar_senal_original_y_filtrada_con_transformada(senal_convulsion, senal_convulsion_f, 'Senal Convulsion filtrada')

    # 3. Potencia por bandas
    # Paso 1: obtener frecuencias y espectro
    xf_sana, espectro_sana= calcular_espectro_frecuencias(senal_sana_f, fs= FRECUENCIA_MUESTREO)
    xf_inter, espectro_inter = calcular_espectro_frecuencias(senal_interictal_f, fs= FRECUENCIA_MUESTREO)
    xf_conv, espectro_conv = calcular_espectro_frecuencias(senal_convulsion_f, fs= FRECUENCIA_MUESTREO)
    
    # Paso 2: calcular la potencia espectral
    psd_sana = calcular_potencia_espectral(espectro_sana, len(senal_sana_f), fs= FRECUENCIA_MUESTREO)
    psd_inter = calcular_potencia_espectral(espectro_inter, len(senal_interictal_f), fs= FRECUENCIA_MUESTREO)
    psd_conv = calcular_potencia_espectral(espectro_conv, len(senal_convulsion_f), fs= FRECUENCIA_MUESTREO)
    
    # Paso 3: calcular la potencia por bandas
    potencias_sana = calcular_potencia_bandas(xf_sana, psd_sana)
    potencias_inter = calcular_potencia_bandas(xf_inter, psd_inter)
    potencias_conv = calcular_potencia_bandas(xf_conv, psd_conv)
    graficar_comparacion_potencias(potencias_sana, potencias_inter, potencias_conv)




    # 4. Autocorrelacion
    lags_sana, autocorr_sana = calcular_autocorrelacion(senal_sana)
    lags_int, autocorr_interictal = calcular_autocorrelacion(senal_interictal)
    lags_conv, autocorr_convulsion = calcular_autocorrelacion(senal_convulsion)
    graficar_autocorrelacion_con_senal_original(senal_sana, autocorr_sana, lags_sana, "Paciente Sano")
    graficar_autocorrelacion_con_senal_original(senal_interictal, autocorr_interictal, lags_int, "Paciente Interictal")
    graficar_autocorrelacion_con_senal_original(senal_convulsion, autocorr_convulsion, lags_conv, "Paciente Convulsión")

    print("\nAnálisis completado.")

# Ejecutar análisis
if __name__ == "__main__":
    analisis_completo_eeg()