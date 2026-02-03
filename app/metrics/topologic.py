import cv2
import numpy as np
from scipy.ndimage import generic_filter

# Bucles, Puntas y Uniones
def get_topology(skel):
    padded = cv2.copyMakeBorder(skel, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    loops = sum(1 for h in hierarchy[0] if h[3] != -1) if hierarchy is not None else 0
    
    def count_neighbors(P):
        return np.sum(P) - 1 if P[4] == 1 else 0

    neighbor_map = generic_filter(skel, count_neighbors, size=(3,3))
    endpoints = int(np.sum(neighbor_map == 1))
    junctions = int(np.sum(neighbor_map >= 3))
    
    return {"loops": loops, "endpoints": endpoints, "junctions": junctions}