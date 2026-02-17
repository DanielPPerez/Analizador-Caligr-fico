import cv2
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage
from app.core.config import MIN_BRANCH_LENGTH, TARGET_SIZE

def clean_notebook_v7(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1. Contraste máximo para rescatar lápiz tenue
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 2. Binarización muy sensible
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 7
    )

    # 3. DETECCIÓN Y ELIMINACIÓN AGRESIVA DE LÍNEAS HORIZONTALES
    # Buscamos estructuras muy anchas (líneas de libreta)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    # Las dilatamos un poco para asegurar que borramos toda la línea
    detected_lines = cv2.dilate(detected_lines, np.ones((3,3), np.uint8))
    # Restamos las líneas
    clean_binary = cv2.subtract(binary, detected_lines)

    # 4. SANADO DE TRAZO (Cerrado morfológico muy fuerte)
    # Esto "suelda" las partes de la letra que se rompieron al quitar las líneas
    kernel_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel_heal)

    # 5. Selección del componente más grande (La Letra)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(clean_binary, connectivity=8)
    if nb_components <= 1: return clean_binary
    
    # Filtrar por área mínima para ignorar basura residual de las líneas
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    final_mask = np.zeros_like(clean_binary)
    if stats[max_label, cv2.CC_STAT_AREA] > 100: # Solo si tiene un tamaño razonable
        final_mask[output == max_label] = 255
    
    return final_mask

def prune_skeleton(skel, min_branch_length=MIN_BRANCH_LENGTH):
    skel = skel.copy().astype(np.uint8)
    def get_neighbors(y, x, img):
        y0, y1 = max(0, y-1), min(img.shape[0], y+2)
        x0, x1 = max(0, x-1), min(img.shape[1], x+2)
        neighborhood = img[y0:y1, x0:x1]
        indices = np.argwhere(neighborhood == 1)
        return [(y0 + i[0], x0 + i[1]) for i in indices if not (y0 + i[0] == y and x0 + i[1] == x)]

    while True:
        changed = False
        neighbor_count = ndimage.generic_filter(skel, lambda P: np.sum(P)-1 if P[4]==1 else 0, size=(3,3), mode='constant')
        endpoints = np.argwhere(neighbor_count == 1)
        for ep in endpoints:
            branch = [tuple(ep)]
            curr = tuple(ep)
            is_spur = False
            for _ in range(min_branch_length):
                neighbors = get_neighbors(curr[0], curr[1], skel)
                next_pts = [n for n in neighbors if n not in branch]
                if not next_pts or len(next_pts) > 1:
                    is_spur = True; break
                curr = next_pts[0]; branch.append(curr)
                if neighbor_count[curr[0], curr[1]] > 2:
                    is_spur = True; break
            if is_spur and len(branch) < min_branch_length:
                for y_b, x_b in branch: skel[y_b, x_b] = 0
                changed = True
        if not changed: break
    return skel

def preprocess_robust(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    binary = clean_notebook_v7(img)
    coords = cv2.findNonZero(binary)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y:y+h, x:x+w]
    
    side = max(w, h) + 30
    square = np.zeros((side, side), dtype=np.uint8)
    off_y, off_x = (side - h) // 2, (side - w) // 2
    square[off_y:off_y+h, off_x:off_x+w] = crop
    
    resized = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    _, binarized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    skel = skeletonize(binarized > 0, method='lee').astype(np.uint8)
    return prune_skeleton(skel)