import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal

# Configuración constante
FS = 173.61  # Frecuencia de muestreo en Hz
ORDEN_FILTRO = 4
FRECUENCIA_CORTE = 40  # Hz para filtro pasa bajos


def filtrar_senal(senal, frecuencia_corte=40, frecuencia_muestreo=173.61, orden_filtro=6):
    """Filtrado con normalización de amplitud"""
    # Remover componente DC
    senal_sin_dc = senal - np.mean(senal)

    # Diseñar filtro
    nyquist = 0.5 * frecuencia_muestreo
    frecuencia_normalizada = frecuencia_corte / nyquist
    b, a = signal.butter(orden_filtro, frecuencia_normalizada, btype='low')

    # Aplicar filtro con fase lineal
    senal_filtrada = signal.filtfilt(b, a, senal_sin_dc)

    # Normalizar amplitud
    senal_filtrada = (senal_filtrada / np.max(np.abs(senal_filtrada))) * np.max(np.abs(senal))

    return senal_filtrada


def calcular_espectro(senal, fs=FS):
    """Calcula la transformada de Fourier de la señal"""
    n = len(senal)
    yf = fft(senal)
    xf = fftfreq(n, 1/fs)[:n//2]
    return xf, 2/n * np.abs(yf[0:n//2])

def calcular_potencia_banda(frecuencias, amplitud, fmin, fmax):
    """Calcula la potencia en una banda específica de frecuencia"""
    mask = (frecuencias >= fmin) & (frecuencias <= fmax)
    return np.trapz(amplitud[mask], frecuencias[mask])

def extraer_caracteristicas(frecuencias, amplitud):
    """Extrae características espectrales relevantes para la clasificación"""
    # Potencias por banda
    delta = calcular_potencia_banda(frecuencias, amplitud, 0.5, 4)
    theta = calcular_potencia_banda(frecuencias, amplitud, 4, 8)
    alpha = calcular_potencia_banda(frecuencias, amplitud, 8, 13)
    beta = calcular_potencia_banda(frecuencias, amplitud, 13, 30)
    gamma = calcular_potencia_banda(frecuencias, amplitud, 30, 40)
    
    # Ratios importantes
    ratio_alpha_theta = alpha / theta if theta != 0 else 0
    ratio_beta_alpha = beta / alpha if alpha != 0 else 0
    ratio_gamma_alpha = gamma / alpha if alpha != 0 else 0
    
    # Ancho de banda espectral
    mascara_amplitud = amplitud > 0.1*np.max(amplitud)
    if np.any(mascara_amplitud):
        ancho_banda = np.max(frecuencias[mascara_amplitud]) - np.min(frecuencias[mascara_amplitud])
    else:
        ancho_banda = 0
    
    return {
        'delta': delta,
        'theta': theta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'ratio_alpha_theta': ratio_alpha_theta,
        'ratio_beta_alpha': ratio_beta_alpha,
        'ratio_gamma_alpha': ratio_gamma_alpha,
        'ancho_banda': ancho_banda,
        'potencia_total': delta + theta + alpha + beta + gamma
    }

def clasificar_senal(senal, umbral_gamma=15, umbral_theta=10, umbral_alpha_theta=1.5):
    
    # Preprocesamiento
    senal_filtrada = filtrar_senal(senal)
    frecuencias, amplitud = calcular_espectro(senal_filtrada)
    
    # Extraer características
    caracteristicas = extraer_caracteristicas(frecuencias, amplitud)
    
    # Inicializar probabilidades base
    prob_sano = 0.3
    prob_interictal = 0.4
    prob_convulsion = 0.3
    
    # Ajustar probabilidades según características
    if caracteristicas['gamma'] > umbral_gamma:
        prob_convulsion += 0.3
        prob_interictal -= 0.15
        prob_sano -= 0.15
    elif caracteristicas['theta'] > umbral_theta:
        prob_interictal += 0.3
        prob_sano -= 0.15
        prob_convulsion -= 0.15
    
    if caracteristicas['ratio_alpha_theta'] > umbral_alpha_theta:
        prob_sano += 0.2
        prob_interictal -= 0.1
        prob_convulsion -= 0.1
    
    # Asegurar que las probabilidades no sean negativas
    prob_sano = max(0, prob_sano)
    prob_interictal = max(0, prob_interictal)
    prob_convulsion = max(0, prob_convulsion)
    
    # Normalizar probabilidades
    total = prob_sano + prob_interictal + prob_convulsion
    if total > 0:
        prob_sano /= total
        prob_interictal /= total
        prob_convulsion /= total
    
    # Determinar estado principal
    if prob_convulsion > 0.65:
        estado = "Convulsión"
    elif prob_interictal > 0.55:
        estado = "Interictal"
    elif prob_sano > 0.55:
        estado = "Sano"
    else:
        # Si ninguna probabilidad es clara, elegir la mayor
        max_prob = max(prob_sano, prob_interictal, prob_convulsion)
        if max_prob == prob_sano:
            estado = "Sano (baja confianza)"
        elif max_prob == prob_interictal:
            estado = "Interictal (baja confianza)"
        else:
            estado = "Convulsión (baja confianza)"
    
    return {
        'estado': estado,
        'caracteristicas': caracteristicas,
        'probabilidades': {
            'Sano': prob_sano,
            'Interictal': prob_interictal,
            'Convulsion': prob_convulsion
        }
    }

def mostrar_resultado(resultado):
    print("\n--- Resultado de Clasificación ---")
    print(f"Estado predicho: {resultado['estado']}")
    print("\nProbabilidades:")
    for estado, prob in resultado['probabilidades'].items():
        print(f"- {estado}: {prob:.2f}")
    
    print("\nCaracterísticas espectrales:")
    print(f"- Delta (0.5-4 Hz): {resultado['caracteristicas']['delta']:.2f}")
    print(f"- Theta (4-8 Hz): {resultado['caracteristicas']['theta']:.2f}")
    print(f"- Alpha (8-13 Hz): {resultado['caracteristicas']['alpha']:.2f}")
    print(f"- Beta (13-30 Hz): {resultado['caracteristicas']['beta']:.2f}")
    print(f"- Gamma (30-40 Hz): {resultado['caracteristicas']['gamma']:.2f}")
    print(f"\nRatio Alpha/Theta: {resultado['caracteristicas']['ratio_alpha_theta']:.2f}")
    print(f"Ratio Gamma/Alpha: {resultado['caracteristicas']['ratio_gamma_alpha']:.2f}")
    print(f"Ancho de banda: {resultado['caracteristicas']['ancho_banda']:.2f} Hz")
    print(f"-----------------------------------------------------------------------------")
# Ejemplo de uso
if __name__ == "__main__":

    # Cargar señal de ejemplo
    señal_ejemplo = np.loadtxt('Signal_1.txt')
    señal_ejemplo2 = np.loadtxt('Signal_2.txt')
    señal_ejemplo3 = np.loadtxt('Signal_3.txt')
        
    resultado = clasificar_senal(señal_ejemplo)
    resultado2 = clasificar_senal(señal_ejemplo2)
    resultado3 = clasificar_senal(señal_ejemplo3)

    mostrar_resultado(resultado)
    mostrar_resultado(resultado2)
    mostrar_resultado(resultado3)