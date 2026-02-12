import io
import base64

import cv2
import matplotlib.pyplot as plt
import numpy as np

from app.core.config import TARGET_SHAPE


def _scale_skel_to_fit_guide(skel_a, skel_p):
    """
    Redimensiona el esqueleto del alumno para que encaje lo más posible en la guía.
    Ajusta escala y centrado según los bounding boxes de ambos.
    """
    h, w = skel_p.shape
    # Bounding boxes
    pts_p = np.argwhere(skel_p > 0)
    pts_a = np.argwhere(skel_a > 0)
    if len(pts_p) == 0 or len(pts_a) == 0:
        return skel_a
    min_p, max_p = pts_p.min(axis=0), pts_p.max(axis=0)
    min_a, max_a = pts_a.min(axis=0), pts_a.max(axis=0)
    size_p = max_p - min_p + 1
    size_a = max_a - min_a + 1
    if size_a[0] < 2 or size_a[1] < 2:
        return skel_a
    scale = min(size_p[0] / size_a[0], size_p[1] / size_a[1])
    center_a = (min_a + max_a) / 2
    center_p = (min_p + max_p) / 2
    M = np.float32([
        [scale, 0, center_p[1] - center_a[1] * scale],
        [0, scale, center_p[0] - center_a[0] * scale]
    ])
    skel_scaled = cv2.warpAffine(
        skel_a.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return (skel_scaled > 0).astype(np.uint8)


def generate_comparison_plot(skel_p, skel_a, score):
    target_size = TARGET_SHAPE
    if skel_p.shape != target_size:
        skel_p = cv2.resize(skel_p.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    if skel_a.shape != target_size:
        skel_a = cv2.resize(skel_a.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    # Escalar letra del alumno para que encaje en la guía (solo para visualización)
    skel_a_display = _scale_skel_to_fit_guide(skel_a, skel_p)
    skel_a = skel_a_display

    h, w = target_size
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Lógica de colores para Debug Profesional:
    # Amarillo (255, 255, 0) -> Coincidencia perfecta (Intersección)
    # Verde (0, 255, 0)      -> Parte de la plantilla que no se tocó
    # Rojo (255, 0, 0)       -> Parte que el alumno dibujó fuera de la plantilla
    
    intersection = np.logical_and(skel_p > 0, skel_a > 0)
    only_p = np.logical_and(skel_p > 0, np.logical_not(skel_a > 0))
    only_a = np.logical_and(skel_a > 0, np.logical_not(skel_p > 0))

    overlay[only_p] = [0, 150, 0]       # Verde oscuro (Guía)
    overlay[only_a] = [255, 50, 50]     # Rojo (Error/Desvío)
    overlay[intersection] = [255, 255, 0] # Amarillo (Acierto)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(overlay)
    ax.set_title(f"Evaluación: {score:.2f}%", color="white", fontsize=12, fontweight="bold")
    legend_y = h - 6
    ax.text(4, legend_y, "Verde: Guía | Rojo: Error | Amarillo: Acierto",
            color="white", fontsize=7, bbox=dict(facecolor="black", alpha=0.5))
    
    ax.axis('off')
    fig.patch.set_facecolor('#1e1e1e') 

    # 5. Convertir gráfica a Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    
    return img_str