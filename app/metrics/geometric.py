import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2

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
    Calcula métricas geométricas entre dos esqueletos.
    
    Args:
        skel_p: Esqueleto de la plantilla (patrón)
        skel_a: Esqueleto del alumno
        tolerance_radius: Radio de tolerancia en píxeles para el IOU (default: 2)
                         Permite pequeñas diferencias de posición sin penalizar tanto
        align: Si True, alinea los esqueletos por centroide antes de calcular métricas
    
    Returns:
        dict con iou, hausdorff y score
    """
    if skel_p.shape != skel_a.shape:
        skel_a = cv2.resize(skel_a.astype(np.uint8), (skel_p.shape[1], skel_p.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Alinear esqueletos si se solicita
    if align:
        skel_p, skel_a = align_skeletons(skel_p, skel_a)
    
    # IOU exacto (píxel a píxel) - para referencia
    intersection_exact = np.logical_and(skel_p > 0, skel_a > 0).sum()
    union_exact = np.logical_or(skel_p > 0, skel_a > 0).sum()
    iou_exact = (intersection_exact / union_exact * 100) if union_exact != 0 else 0.0
    
    # IOU con tolerancia espacial (dilatación)
    # Esto permite que píxeles cercanos se consideren como coincidentes
    if tolerance_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance_radius+1, 2*tolerance_radius+1))
        skel_p_dilated = cv2.dilate((skel_p > 0).astype(np.uint8), kernel, iterations=1)
        skel_a_dilated = cv2.dilate((skel_a > 0).astype(np.uint8), kernel, iterations=1)
        
        intersection_tol = np.logical_and(skel_p_dilated > 0, skel_a_dilated > 0).sum()
        union_tol = np.logical_or(skel_p_dilated > 0, skel_a_dilated > 0).sum()
        iou = (intersection_tol / union_tol * 100) if union_tol != 0 else 0.0
    else:
        iou = iou_exact
    
    # Calcular Hausdorff distance
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
        "iou_exact": float(round(iou_exact, 2)),  # Para debug
        "hausdorff": float(round(haus_dist, 2)), 
        "score": float(round(score, 2))
    }