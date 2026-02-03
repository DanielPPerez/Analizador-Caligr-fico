
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def _normalize_size(skel, target_size=256):
    """Redimensiona el esqueleto a target_size x target_size si no coincide."""
    if skel.shape[0] == target_size and skel.shape[1] == target_size:
        return skel
    resized = cv2.resize(
        skel.astype(np.uint8),
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8)

def calculate_geometric(skel_p, skel_a):
    # Normalizar tamaños para poder comparar (ej. plantilla 200x200 vs alumno 256x256)
    TARGET = 256
    skel_p = _normalize_size(skel_p, TARGET)
    skel_a = _normalize_size(skel_a, TARGET)

    intersection = np.logical_and(skel_p, skel_a).sum()
    union = np.logical_or(skel_p, skel_a).sum()
    iou = (intersection / union) * 100 if union != 0 else 0.0
    
    pts_p = np.argwhere(skel_p > 0)
    pts_a = np.argwhere(skel_a > 0)
    
    if len(pts_p) == 0 or len(pts_a) == 0:
        haus_dist = 999.0
    else:
        d1 = directed_hausdorff(pts_p, pts_a)[0]
        d2 = directed_hausdorff(pts_a, pts_p)[0]
        haus_dist = max(d1, d2)

    if np.isinf(haus_dist) or np.isnan(haus_dist):
        haus_dist = 999.0

    score = max(0.0, 100.0 - (float(haus_dist) * 1.5))
    
    return {
        "iou": float(round(iou, 2)), 
        "hausdorff": float(round(haus_dist, 2)), 
        "score": float(round(score, 2))
    }