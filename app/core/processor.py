import cv2
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage
import os

# Importamos configuraciones (Asegúrate de que en config.py TARGET_SHAPE sea (128, 128))
try:
    from app.core.config import TARGET_SHAPE, MIN_BRANCH_LENGTH
except ImportError:
    TARGET_SHAPE = (128, 128)
    MIN_BRANCH_LENGTH = 22

def prune_skeleton(skel, min_branch_length=None):
    """
    Elimina ramas recursivamente para limpiar el esqueleto de 'vellosidades'.
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
        neighbor_count = ndimage.generic_filter(
            skel, lambda P: np.sum(P) - 1 if P[4] == 1 else 0, size=(3, 3), mode='constant'
        )
        
        endpoints = np.argwhere(neighbor_count == 1)
        for ep in endpoints:
            branch = []
            curr = tuple(ep)
            is_spur = False
            
            for _ in range(min_branch_length):
                branch.append(curr)
                neighbors = get_neighbors(curr[0], curr[1], skel)
                next_pts = [n for n in neighbors if n not in branch]
                
                if len(next_pts) == 0:
                    is_spur = True
                    break
                if len(next_pts) > 1:
                    is_spur = True
                    break
                
                curr = next_pts[0]
                if neighbor_count[curr[0], curr[1]] > 2:
                    is_spur = True
                    break
            
            if is_spur and len(branch) < min_branch_length:
                for y_b, x_b in branch:
                    skel[y_b, x_b] = 0
                changed = True
        
        if not changed: break
    return skel

def remove_notebook_lines(binary_img):
    """
    Elimina líneas de libreta/cuaderno (horizontales y verticales) que aparecen
    como artefactos en fotos de celular. Las líneas son largas y delgadas;
    los trazos de letras son más cortos o tienen diferente proporción.
    """
    # Kernel horizontal: detecta líneas horizontales largas (típicas de libretas)
    # Solo estructuras ≥30px de ancho y ≤2px de alto serán extraídas
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, h_kernel)
    binary_img = cv2.subtract(binary_img, horizontal_lines)

    # Kernel vertical: detecta líneas verticales (hojas con márgenes/cuadrícula)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 35))
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, v_kernel)
    binary_img = cv2.subtract(binary_img, vertical_lines)

    return binary_img


def clean_artifacts(binary_img):
    """
    Aisla la letra principal eliminando:
    - Líneas de libreta/cuaderno
    - Bordes de papel, sombras
    - Trazos accidentales en la periferia
    """
    # 0. Eliminar líneas de libreta PRIMERO (antes de componentes conexos)
    binary_img = remove_notebook_lines(binary_img.copy())

    # 1. Limpieza inicial suave (ruido puntual)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 2. Etiquetado de componentes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    if num_labels <= 1: return binary_img

    h_img, w_img = binary_img.shape
    img_center = np.array([h_img / 2, w_img / 2])
    
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # REGLA 1: Ignorar lo que toca los bordes extremos (bordes de papel o sombras de orilla)
        is_border = (left <= 2 or top <= 2 or (left + w) >= w_img - 2 or (top + h) >= h_img - 2)

        # REGLA 1.5: Descartar fragmentos de líneas (aspecto muy alargado)
        # Líneas de libreta: relación ancho/alto > 8 o alto/ancho > 8
        aspect = max(w, h) / (min(w, h) + 1e-6)
        is_line_fragment = (aspect > 8) or (area < 500 and aspect > 5)
        
        # REGLA 2: Área mínima para ser considerada una letra
        if area > 100 and not is_border and not is_line_fragment:
            # Calculamos distancia al centro para dar prioridad a la letra principal
            dist = np.linalg.norm(centroids[i][::-1] - img_center)
            candidates.append((i, area, dist))

    if not candidates:
        # Si todo fue filtrado, tomamos el componente más grande que no sea fondo
        # (caso de emergencia para no devolver imagen vacía)
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = np.zeros_like(binary_img)
        mask[labels == largest] = 255
        return mask

    # Elegimos el candidato con mejor relación Área/Proximidad al centro
    # En caligrafía, la letra suele ser el objeto más grande cerca del centro
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_label = candidates[0][0]
    
    final_mask = np.zeros_like(binary_img)
    final_mask[labels == best_label] = 255
    return final_mask

def preprocess_robust(img_bytes):
    """
    Pipeline robusto para fotos de celular: 
    Extracción -> Limpieza de Artefactos -> Normalización 128x128 -> Esqueleto.
    """
    # 1. Decodificar
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # --- FASE 1: FILTRADO DE FONDO (Black Hat) ---
    # Elimina la iluminación desigual y el fondo del papel
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_bh)
    
    # --- FASE 2: NORMALIZACIÓN DE CONTRASTE ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blackhat)

    # --- FASE 3: BINARIZACIÓN ---
    # Otsu funciona bien con fondos uniformes; si hay líneas de libreta, el umbral puede fallar.
    # Probamos Otsu primero; si el resultado tiene demasiado ruido (muchos componentes pequeños),
    # usamos umbral adaptativo como fallback.
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_comp = cv2.connectedComponentsWithStats(binary, connectivity=8)[0]
    if num_comp > 50:  # Muchos componentes = probable ruido/líneas mal binarizadas
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 4
        )

    # --- FASE 4: LIMPIEZA DE ARTEFACTOS (Líneas libreta, papel, manchas) ---
    clean_binary = clean_artifacts(binary)

    # --- FASE 5: NORMALIZACIÓN A 128x128 (Manteniendo proporción) ---
    coords = cv2.findNonZero(clean_binary)
    if coords is None: return None
    
    x, y, w, h = cv2.boundingRect(coords)
    crop = clean_binary[y:y+h, x:x+w]
    
    # Definimos el lienzo final (128x128)
    # Margen reducido para maximizar tamaño de letra (encaje en guía)
    side = 128
    margin = 6  # Mínimo respiro para que la letra ocupe casi todo el lienzo
    inner_side = side - (margin * 2)
    
    # Redimensionar conservando aspect ratio (Letterboxing)
    scale = inner_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_char = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pegar en el centro del lienzo negro
    canvas = np.zeros((side, side), dtype=np.uint8)
    start_x = (side - new_w) // 2
    start_y = (side - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_char

    # --- FASE 6: RE-BINARIZACIÓN Y SUAVIZADO ---
    # Esto prepara la imagen para que el esqueleto sea suave (ideal para ResNet)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, ready_for_skel = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)

    # --- FASE 7: ESQUELETIZACIÓN ---
    skel = skeletonize(ready_for_skel > 0, method='lee').astype(np.uint8)
    
    # Poda final de ramas parásitas
    return prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH)

# =============================================================================
# PREPARADO PARA RESNET Y OCR (próxima implementación)
# =============================================================================
# get_image_for_resnet(img_bytes) -> np.ndarray 128x128 binaria, lista para
#   modelo de clasificación (ResNet). Usar ready_for_skel antes de esqueletizar.
# classify_char(resnet_model, img) -> char  # Futuro: OCR/clasificación
# =============================================================================

def get_image_for_resnet(img_bytes):
    """
    Devuelve la imagen preprocesada lista para ResNet/OCR (128x128, binaria).
    Usar cuando se implemente el clasificador de caracteres.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_bh)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blackhat)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_comp = cv2.connectedComponentsWithStats(binary, connectivity=8)[0]
    if num_comp > 50:
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 4
        )
    clean_binary = clean_artifacts(binary)
    coords = cv2.findNonZero(clean_binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = clean_binary[y:y+h, x:x+w]
    side, margin = 128, 6
    inner_side = side - (margin * 2)
    scale = inner_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_char = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side), dtype=np.uint8)
    start_x, start_y = (side - new_w) // 2, (side - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_char
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, ready = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
    return ready