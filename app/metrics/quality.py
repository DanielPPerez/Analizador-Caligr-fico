import cv2
import numpy as np

def calculate_quality_metrics(skel_a):
    """Evalúa aspectos de forma y proporción del trazo."""
    coords = cv2.findNonZero(skel_a)
    if coords is None: return {"aspect_ratio": 0, "hu_moments": []}

    # 1. Relación de Aspecto (Aspect Ratio)
    x, y, w, h = cv2.boundingRect(coords)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    # 2. Momentos de Hu (Firma geométrica)
    moments = cv2.moments(skel_a)
    hu = cv2.HuMoments(moments).flatten()
    # Transformación logarítmica para que los números sean comparables
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-15)

    return {
        "aspect_ratio": round(aspect_ratio, 2),
        "shape_fingerprint": hu_log.tolist()[:3] # Los 3 más significativos
    }