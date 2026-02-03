import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Tamaño común para la comparación visual
PLOT_SIZE = 256

def _same_size(skel: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona el esqueleto a size x size si no lo es ya."""
    if skel.shape[0] == size and skel.shape[1] == size:
        return skel
    resized = cv2.resize(
        skel.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST
    )
    return (resized > 0).astype(np.uint8)

def generate_comparison_plot(skel_p: np.ndarray, skel_a: np.ndarray, score: float) -> str:
    """
    Genera una imagen de comparación patrón vs alumno en base64.
    Normaliza ambos esqueletos al mismo tamaño para evitar errores de shape.
    """
    p = _same_size(skel_p, PLOT_SIZE)
    a = _same_size(skel_a, PLOT_SIZE)

    overlay = np.zeros((PLOT_SIZE, PLOT_SIZE, 3), dtype=np.uint8)
    overlay[p > 0] = [0, 255, 0]   # Patrón verde
    overlay[a > 0] = [255, 0, 0]   # Alumno rojo

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.set_title(f"Resultado Final - Score: {score}")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

