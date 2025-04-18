# main.py

# Importación de bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from env import VISUAL_CONFIG

# Aplicar configuración visual
plt.rcParams.update(VISUAL_CONFIG)

from funciones_eeg import (
    cargar_senales, filtrar_senal, graficar_comparacion_tiempo,
    analizar_distribucion_bandas, calcular_espectro_frecuencias,
    graficar_espectro_frecuencias, calcular_potencia_bandas,
    graficar_comparacion_potencias, calcular_autocorrelacion,
    graficar_autocorrelaciones, graficar_senales_tiempo,
    graficar_autocorrelacion_con_senal_original
)
# =============================================
# Función principal de análisis
# =============================================

def analisis_completo_eeg():
    # 1. Carga y filtrado de señales
    senal_sana, senal_interictal, senal_convulsion = cargar_senales()
    
    # Aplicar filtro
    senal_sana_f = filtrar_senal(senal_sana)
    senal_interictal_f = filtrar_senal(senal_interictal)
    senal_convulsion_f = filtrar_senal(senal_convulsion)

    # Graficar comparación original vs filtrado
    print("\nComparación señales originales vs filtradas...")
    # graficar_comparacion_tiempo(senal_sana, senal_sana_f, 'Señal Sana')
    # graficar_comparacion_tiempo(senal_interictal, senal_interictal_f, 'Señal Interictal')
    # graficar_comparacion_tiempo(senal_convulsion, senal_convulsion_f, 'Señal de Convulsión')
    # graficar_senal_y_transformada(senal_sana_f, 'Senal Sana filtrada')
    # graficar_senal_y_transformada(senal_interictal_f, 'Senal Interictal filtrada')
    # graficar_senal_y_transformada(senal_convulsion_f, 'Senal Convulsion filtrada')

    # 2. Análisis espectral
    # print("\nAnalizando distribución espectral...")
    # espectros = {
    #     'Sana': (senal_sana_f, 'Señal Sana'),
    #     'Interictal': (senal_interictal_f, 'Señal Interictal'),
    #     'Convulsion': (senal_convulsion_f, 'Señal Convulsión')
    # }
    #
    # for clave, (senal, nombre) in espectros.items():
    #     xf, yf = calcular_espectro_frecuencias(senal)
    #     print(f"\n==== Análisis detallado - {nombre} ====")
    #     analizar_distribucion_bandas(xf, yf)
    #     graficar_espectro_frecuencias(xf, yf, nombre)



    # 3. Potencia por bandas (opcional)
    # print("\nCalculando potencia por bandas espectrales...")
    # xf_sana, yf_sana = calcular_espectro_frecuencias(senal_sana_f)
    # xf_inter, yf_inter = calcular_espectro_frecuencias(senal_interictal_f)
    # xf_conv, yf_conv = calcular_espectro_frecuencias(senal_convulsion_f)
    
    # potencias_sana = calcular_potencia_bandas(xf_sana, yf_sana)
    # potencias_inter = calcular_potencia_bandas(xf_inter, yf_inter)
    # potencias_conv = calcular_potencia_bandas(xf_conv, yf_conv)

    # graficar_comparacion_potencias(potencias_sana, potencias_inter, potencias_conv)



    # 4. Autocorrelación (opcional)
    # print("\nCalculando autocorrelaciones...")
    # autocorr_sana = calcular_autocorrelacion(senal_sana_f)
    # autocorr_inter = calcular_autocorrelacion(senal_interictal_f)
    # autocorr_conv = calcular_autocorrelacion(senal_convulsion_f)
    # graficar_autocorrelaciones(autocorr_sana, autocorr_inter, autocorr_conv)    
    # graficar_autocorrelacion_con_senal_original(senal_sana_f, autocorr_sana, 'Señal Sana')
    # graficar_autocorrelacion_con_senal_original(senal_interictal_f, autocorr_inter, 'Señal Interictal')
    # graficar_autocorrelacion_con_senal_original(senal_convulsion_f, autocorr_conv, 'Señal Convulsión')
    lags_sana, autocorr_sana = calcular_autocorrelacion(senal_sana)
    lags_int, autocorr_interictal = calcular_autocorrelacion(senal_interictal)
    lags_conv, autocorr_convulsion = calcular_autocorrelacion(senal_convulsion)

# Graficar individual
    graficar_autocorrelacion_con_senal_original(senal_sana, autocorr_sana, lags_sana, "Paciente Sano")
    graficar_autocorrelacion_con_senal_original(senal_interictal, autocorr_interictal, lags_int, "Paciente Interictal")
    graficar_autocorrelacion_con_senal_original(senal_convulsion, autocorr_convulsion, lags_conv, "Paciente Convulsión")


    print("\nAnálisis completado.")

# Ejecutar análisis
if __name__ == "__main__":
    analisis_completo_eeg()