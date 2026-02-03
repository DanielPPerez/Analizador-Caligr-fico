import cv2
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage

def prune_skeleton(skel, min_branch_length=30):
    """
    Elimina ramas cortas que salen de intersecciones (común en M, N, W).
    """
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
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # Binarización y limpieza inicial
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary = cv2.medianBlur(binary, 3)
    
    coords = cv2.findNonZero(binary)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y:y+h, x:x+w]
    
    # Normalización a 256x256 con margen
    side = max(w, h) + 60
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side-h)//2 : (side-h)//2 + h, (side-w)//2 : (side-w)//2 + w] = crop
    resized = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Suavizado para eliminar esquinas cuadradas pre-esqueleto
    resized = cv2.GaussianBlur(resized, (9, 9), 0)
    _, resized = cv2.threshold(resized, 110, 255, cv2.THRESH_BINARY)

    # Esqueletización y Poda
    skel = skeletonize(resized > 0, method='lee').astype(np.uint8)
    return prune_skeleton(skel, min_branch_length=20)