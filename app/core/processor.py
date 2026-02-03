import cv2
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndimage

def prune_skeleton(skel, min_branch_length=20):
    """
    Poda ramas muertas preservando la estructura principal.
    """
    skel = skel.copy()
    
    def count_neighbors(P):
        # Cuenta vecinos en una vecindad de 3x3
        return np.sum(P) - 1 if P[4] == 1 else 0

    iteration_limit = 5
    for _ in range(iteration_limit):
        changed = False
        neighbor_map = ndimage.generic_filter(skel, count_neighbors, size=(3, 3), mode='constant')
        
        # Encontrar puntas (píxeles con solo 1 vecino)
        endpoints = np.argwhere(neighbor_map == 1)

        for ep in endpoints:
            branch = []
            curr = tuple(ep)
            
            # Rastrear la rama desde la punta hasta una intersección
            while True:
                branch.append(curr)
                y, x = curr
                
                # Buscar el siguiente píxel conectado
                y0, y1 = max(0, y-1), min(skel.shape[0], y+2)
                x0, x1 = max(0, x-1), min(skel.shape[1], x+2)
                
                neighborhood = skel[y0:y1, x0:x1]
                neighbors = np.argwhere(neighborhood == 1)
                
                next_pts = []
                for n in neighbors:
                    nxt = (y0 + n[0], x0 + n[1])
                    if nxt not in branch:
                        next_pts.append(nxt)
                
                # Si llegamos a una intersección (>1 vecino) o callejón sin salida
                if len(next_pts) != 1:
                    break
                curr = next_pts[0]
            
            # Si la rama es muy corta, la borramos
            if len(branch) < min_branch_length:
                for y_b, x_b in branch:
                    skel[y_b, x_b] = 0
                changed = True
        
        if not changed:
            break
            
    return skel

def preprocess_robust(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # 1. Binarización
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary = cv2.medianBlur(binary, 3)
    
    # 2. Rotación 
    coords = cv2.findNonZero(binary)
    if coords is None: return None
    
    # 3. CREAR LIENZO CUADRADO 
    x, y, w_c, h_c = cv2.boundingRect(coords)
    crop = binary[y:y+h_c, x:x+w_c]
    
    side = max(w_c, h_c) + 40 
    square_canvas = np.zeros((side, side), dtype=np.uint8)
    off_y = (side - h_c) // 2
    off_x = (side - w_c) // 2
    square_canvas[off_y:off_y+h_c, off_x:off_x+w_c] = crop

    # 4. REDIMENSIONAR A 256x256
    resized = cv2.resize(square_canvas, (256, 256), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    # 5. Esqueleto y poda 
    skel = skeletonize(resized > 0).astype(np.uint8)
    skel = prune_skeleton(skel, min_branch_length=15)
    
    return skel