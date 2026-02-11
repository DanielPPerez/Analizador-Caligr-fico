"""
Configuración compartida: resolución de esqueletos y poda.
Todas las métricas y el preprocesamiento usan 128×128 para menor uso de memoria.
"""
# Resolución objetivo de los esqueletos (plantillas y alumno)
TARGET_SIZE = 128
TARGET_SHAPE = (TARGET_SIZE, TARGET_SIZE)

# Poda de espolones: misma longitud mínima para plantillas y alumno (pipeline alineado)
# A 128×128 las ramas son ~mitad de píxeles que a 256×256, usamos 10 para proporción similar
MIN_BRANCH_LENGTH = 10

# Submuestreo de puntos para trayectoria/Procrustes (reducir N, M)
MAX_POINTS_TRAJECTORY = 64
PROCRUSTES_N_POINTS = 50

# DTW: ancho de banda (Sakoe-Chiba) para no materializar matriz completa
DTW_BAND_RATIO = 0.25  # banda = ratio * max(n,m)

# Tolerancia en píxeles para score Hausdorff (escala con resolución 128)
HAUSDORFF_TOLERANCE = 5
HAUSDORFF_FACTOR = 2.0  # score = max(0, 100 - (haus - tol) * factor)
