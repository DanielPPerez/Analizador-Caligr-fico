import numpy as np
import cv2
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim

from app.core.config import PROCRUSTES_N_POINTS


def _resample_sequence_to_n(points, n):
    """Remuestrea una secuencia de puntos a exactamente n puntos por interpolación lineal."""
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    if len(points) == 1:
        return np.tile(points, (n, 1))
    # Cerrar el contorno para caracteres con bucles 
    pts = np.vstack([points, points[0]])
    cumlen = np.zeros(len(pts))
    cumlen[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total = cumlen[-1]
    if total == 0:
        return np.tile(points.mean(axis=0), (n, 1))
    target = np.linspace(0, total * (n - 1) / n, n, endpoint=False)
    idx = np.searchsorted(cumlen, target, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    t = (target - cumlen[idx]) / (cumlen[idx + 1] - cumlen[idx] + 1e-9)
    return (1 - t)[:, None] * pts[idx] + t[:, None] * pts[idx + 1]


def calculate_procrustes(skel_p, skel_a, seq_p, seq_a):
    """
    Alineación Procrustes: escala, rotación y traslación óptimas para minimizar
    la suma de cuadrados de diferencias.
    """
    if len(seq_p) < 3 or len(seq_a) < 3:
        return 999.0, 0.0  
    pts_p = _resample_sequence_to_n(seq_p, PROCRUSTES_N_POINTS)
    pts_a = _resample_sequence_to_n(seq_a, PROCRUSTES_N_POINTS)
    try:
        mtx1, mtx2, disparity = procrustes(pts_p, pts_a)
    except (ValueError, np.linalg.LinAlgError):
        return 999.0, 0.0
    score_proc = max(0.0, 100.0 - disparity * 50.0)
    return float(round(disparity, 4)), float(round(score_proc, 2))


def align_skeletons(skel_p, skel_a):
    """
    Alinea dos esqueletos usando sus centroides para compensar desplazamientos.
    Esto ayuda cuando hay pequeñas diferencias de posición debido al preprocesamiento.
    """
    pts_p = np.argwhere(skel_p > 0)
    pts_a = np.argwhere(skel_a > 0)
    
    if len(pts_p) == 0 or len(pts_a) == 0:
        return skel_p, skel_a
    
    # Calcular centroides
    centroid_p = pts_p.mean(axis=0)
    centroid_a = pts_a.mean(axis=0)
    
    # Calcular desplazamiento
    offset = centroid_p - centroid_a
    
    # Si el desplazamiento es muy pequeño (< 1 píxel), no alinear
    if np.linalg.norm(offset) < 1.0:
        return skel_p, skel_a
    
    # Crear matriz de traslación
    rows, cols = skel_a.shape
    M = np.float32([[1, 0, offset[1]], [0, 1, offset[0]]])
    
    # Aplicar traslación al esqueleto del alumno
    skel_a_aligned = cv2.warpAffine(skel_a.astype(np.uint8), M, (cols, rows), 
                                     flags=cv2.INTER_NEAREST, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=0)
    
    return skel_p, skel_a_aligned.astype(skel_p.dtype)

def calculate_geometric(skel_p, skel_a, tolerance_radius=2, align=True):
    """
    Calcula métricas geométricas entre dos esqueletos: SSIM (en lugar de IoU),
    Procrustes (forma global, más tolerante) y Hausdorff (complementario).
    """
    from app.metrics.trajectory import get_sequence_from_skel

    if skel_p.shape != skel_a.shape:
        skel_a = cv2.resize(skel_a.astype(np.uint8), (skel_p.shape[1], skel_p.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if align:
        skel_p, skel_a = align_skeletons(skel_p, skel_a)
    
    # SSIM (Structural Similarity Index)
    img_p = (skel_p > 0).astype(np.uint8)
    img_a = (skel_a > 0).astype(np.uint8)
    ssim_val = ssim(img_p, img_a, data_range=1)
    if np.isnan(ssim_val):
        ssim_val = 0.0
    # SSIM en [-1, 1] -> score 0-100
    ssim_score = float(round((ssim_val + 1) / 2 * 100, 2))
    ssim_val = float(round(ssim_val, 4))
    
    # Secuencias ordenadas para Procrustes
    seq_p = get_sequence_from_skel(skel_p)
    seq_a = get_sequence_from_skel(skel_a)
    procrustes_disparity, procrustes_score = calculate_procrustes(skel_p, skel_a, seq_p, seq_a)
    
    # Hausdorff 
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
    score_hausdorff = max(0.0, 100.0 - (float(haus_dist) * 1.5))
    
    return {
        "ssim": ssim_val,
        "ssim_score": ssim_score,
        "procrustes_disparity": procrustes_disparity,
        "procrustes_score": procrustes_score,
        "hausdorff": float(round(haus_dist, 2)),
        "score": float(round(score_hausdorff, 2)), 
    }