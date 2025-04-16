# env.py

# =============================================
# Configuración de parámetros de procesamiento
# =============================================
FRECUENCIA_CORTE = 20     # Frecuencia de corte para filtro pasa bajos (Hz)
FRECUENCIA_MUESTREO = 210   # Frecuencia de muestreo de las señales (Hz)

# =============================================
# Configuración de visualización
# =============================================
VISUAL_CONFIG = {
    'figure.figsize': [12, 8],
    'font.size': 12
}
# =============================================
# Parámetros de análisis espectral
# =============================================
BANDAS_EEG = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

LIMITE_FRECUENCIAS = 30  # Límite superior para gráficos de frecuencia (Hz)
MAX_RETARDO = 200      # Máximo retardo para autocorrelación (muestras)

