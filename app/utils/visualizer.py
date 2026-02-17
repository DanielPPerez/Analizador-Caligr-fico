import io
import base64
import cv2
import matplotlib.pyplot as plt
import numpy as np
from app.core.config import TARGET_SHAPE

def _scale_skel_to_fit_guide(skel_a, skel_p):
    h, w = skel_p.shape
    pts_p, pts_a = np.argwhere(skel_p > 0), np.argwhere(skel_a > 0)
    if not len(pts_a) or not len(pts_p): return np.zeros_like(skel_p)
    
    min_p, max_p = pts_p.min(axis=0), pts_p.max(axis=0)
    min_a, max_a = pts_a.min(axis=0), pts_a.max(axis=0)
    
    scale = min((max_p[0]-min_p[0]+1)/(max_a[0]-min_a[0]+1), 
                (max_p[1]-min_p[1]+1)/(max_a[1]-min_a[1]+1))
    cp, ca = (min_p + max_p)/2, (min_a + max_a)/2

    M = np.float32([[scale, 0, cp[1] - ca[1]*scale], [0, scale, cp[0] - ca[0]*scale]])
    # Usamos INTER_LINEAR para suavizar el movimiento
    return (cv2.warpAffine(skel_a.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR) > 0).astype(np.uint8)

def generate_comparison_plot(skel_p, skel_a, score):
    h, w = TARGET_SHAPE
    skel_a_scaled = _scale_skel_to_fit_guide(skel_a, skel_p)

    # 1. Crear el overlay a 128x128 primero para lógica de color
    # Usamos dilatación elíptica (redonda) para evitar esquinas cuadradas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    p_thick = cv2.dilate(skel_p.astype(np.uint8), k)
    a_thick = cv2.dilate(skel_a_scaled.astype(np.uint8), k)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    intersection = np.logical_and(p_thick > 0, a_thick > 0)
    only_p = np.logical_and(p_thick > 0, np.logical_not(a_thick > 0))
    only_a = np.logical_and(a_thick > 0, np.logical_not(p_thick > 0))

    overlay[only_p] = [0, 180, 0]          # Verde
    overlay[only_a] = [255, 40, 40]        # Rojo
    overlay[intersection] = [255, 255, 0]  # Amarillo

    # 2. SUAVIZADO MAESTRO: Upscale a 512 + Gaussian Blur ligero
    # Esto elimina los píxeles cuadrados y da un aspecto de caligrafía suave
    overlay_res = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_CUBIC)
    overlay_res = cv2.GaussianBlur(overlay_res, (3, 3), 0.5)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(overlay_res)
    
    # Título estilizado
    ax.set_title(f"Evaluación: {score:.2f}%", color="white", fontsize=16, fontweight="bold", pad=15)
    
    # Leyenda mejor posicionada
    ax.text(256, 495, "Verde: Guía | Rojo: Error | Amarillo: Acierto",
            color="white", fontsize=9, ha='center', fontweight='bold',
            bbox=dict(facecolor="black", alpha=0.7, edgecolor='none', pad=5))
    
    ax.axis('off')
    fig.patch.set_facecolor('#080808')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')