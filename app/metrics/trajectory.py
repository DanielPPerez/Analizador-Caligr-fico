import numpy as np
from scipy.spatial.distance import cdist

def get_sequence_from_skel(skel):
    """Convierte el esqueleto en una lista de puntos (x,y) ordenados por ángulo."""
    points = np.argwhere(skel > 0)
    if len(points) == 0:
        return np.array([])
    
    #ángulo respecto al centro para simular un trazo
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,0] - center[0], points[:,1] - center[1])
    return points[np.argsort(angles)]

def calculate_trajectory_dist(skel_p, skel_a):
    """Calcula la similitud de la trayectoria (DTW simplificado)."""
    seq_p = get_sequence_from_skel(skel_p)
    seq_a = get_sequence_from_skel(skel_a)

    if len(seq_p) == 0 or len(seq_a) == 0:
        return 999.0

    # Distancia promedio mínima entre secuencias
    distances = cdist(seq_p, seq_a, 'euclidean')
    dtw_dist = np.mean(np.min(distances, axis=1))
    
    if np.isnan(dtw_dist) or np.isinf(dtw_dist):
        return 999.0
        
    return float(round(dtw_dist, 2))