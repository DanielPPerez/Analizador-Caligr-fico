import cv2
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage

from app.core.config import TARGET_SHAPE, MIN_BRANCH_LENGTH


def prune_skeleton(skel, min_branch_length=None):
    """
    Elimina ramas cortas que salen de intersecciones (común en M, N, W).
    """
    if min_branch_length is None:
        min_branch_length = MIN_BRANCH_LENGTH
    skel = skel.copy().astype(np.uint8)
    
    def get_neighbors(y, x, img):
        y0, y1 = max(0, y-1), min(img.shape[0], y+2)
        x0, x1 = max(0, x-1), min(img.shape[1], x+2)
        neighborhood = img[y0:y1, x0:x1]
        indices = np.argwhere(neighborhood == 1)
        return [(y0 + i[0], x0 + i[1]) for i in indices if not (y0 + i[0] == y and x0 + i[1] == x)]

    while True:
        changed = False
        # Contar vecinos de cada píxel
        # 1 vecino = punta (endpoint)
        # >2 vecinos = cruce (junction)
        neighbor_count = ndimage.generic_filter(
            skel, lambda P: np.sum(P) - 1 if P[4] == 1 else 0, size=(3, 3), mode='constant'
        )
        
        endpoints = np.argwhere(neighbor_count == 1)
        
        for ep in endpoints:
            branch = []
            curr = tuple(ep)
            is_spur = False
            
            # Rastrear la rama desde la punta hacia adentro
            for _ in range(min_branch_length):
                branch.append(curr)
                neighbors = get_neighbors(curr[0], curr[1], skel)
                next_pts = [n for n in neighbors if n not in branch]
                
                if len(next_pts) == 0:
                    # Rama aislada, se borra
                    is_spur = True
                    break
                if len(next_pts) > 1:
                    # Encontramos una intersección (Junction)
                    is_spur = True
                    break
                
                curr = next_pts[0]
                # Si el punto actual es una intersección (según el mapa original)
                if neighbor_count[curr[0], curr[1]] > 2:
                    is_spur = True
                    break
            
            # Si la rama termina en una intersección y es corta, es un "spur"
            if is_spur and len(branch) < min_branch_length:
                for y_b, x_b in branch:
                    skel[y_b, x_b] = 0
                changed = True
        
        # Si en una pasada no hubo cambios, el esqueleto está limpio
        if not changed:
            break
            
    return skel

def preprocess_robust(img_bytes):
    """
    Preprocesa una imagen del alumno usando un pipeline similar al de las plantillas
    para minimizar diferencias de parametrización.
    
    Detecta si la imagen ya está binarizada (como PNGs de plantillas) y la procesa
    de manera más conservadora.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # Detectar si la imagen ya está binarizada (como los PNGs de plantillas)
    # Una imagen binarizada tiene principalmente valores 0 y 255, con pocos valores intermedios
    unique_vals = np.unique(img)
    binary_ratio = len(unique_vals) / 256.0  # Ratio de valores únicos
    is_likely_binary = binary_ratio < 0.15 or (len(unique_vals) <= 15 and len(unique_vals) >= 2 and np.max(unique_vals) - np.min(unique_vals) > 200)
    
    if is_likely_binary:
        # Imagen ya binarizada (probablemente un PNG de esqueleto)
        # Los PNGs de plantillas tienen fondo negro (0) y esqueleto blanco (255)
        # Verificar si es un esqueleto delgado ANTES de procesar
        active_pixels_orig = np.sum(img > 0)
        total_pixels = img.shape[0] * img.shape[1]
        is_skeleton = active_pixels_orig < total_pixels * 0.1  # Menos del 10% de píxeles activos
        
        if is_skeleton:
            # Es un esqueleto: los PNGs tienen fondo negro (0) y esqueleto blanco (255)
            # skeletonize() busca valores > 0 (True), así que funciona con fondo negro y trazo blanco
            # NO invertir, usar directamente
            binary = img.copy()  # Mantener: fondo negro, trazo blanco
        else:
            # Imagen binaria pero no esqueleto (más gruesa)
            # Determinar orientación y procesar
            if np.mean(img) > 127:  # Más blanco que negro -> fondo claro
                binary = cv2.bitwise_not(img)  # Invertir para tener fondo negro
            else:
                binary = img.copy()
            binary = cv2.medianBlur(binary, 3)
    else:
        # Imagen de foto: procesamiento normal
        try:
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            binary = cv2.medianBlur(binary, 3)
        except:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Si es un esqueleto y ya tiene el tamaño correcto, procesar mínimamente
    if is_likely_binary and is_skeleton and binary.shape == TARGET_SHAPE:
        skel = (binary > 0).astype(np.uint8)
        return prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH)
    
    # Procesamiento normal: recortar, centrar y redimensionar
    coords = cv2.findNonZero(binary)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y:y+h, x:x+w]
    
    # Normalización a TARGET_SIZE con margen (alineado con generate_templates)
    side = max(w, h) + 60
    square = np.zeros((side, side), dtype=np.uint8)
    off_y, off_x = (side - h) // 2, (side - w) // 2
    square[off_y:off_y+h, off_x:off_x+w] = crop

    if is_likely_binary and is_skeleton:
        resized = cv2.resize(square, TARGET_SHAPE, interpolation=cv2.INTER_NEAREST)
        skel = (resized > 0).astype(np.uint8)
        return prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH)
    else:
        resized = cv2.resize(square, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        # Mismo blur y umbral que plantillas: GaussianBlur (9,9), threshold 110
        resized = cv2.GaussianBlur(resized, (9, 9), 0)
        _, resized = cv2.threshold(resized, 110, 255, cv2.THRESH_BINARY)
        skel = skeletonize(resized > 0, method='lee').astype(np.uint8)
        return prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH)