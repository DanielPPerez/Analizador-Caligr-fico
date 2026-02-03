def calculate_final_score(geo_metrics, topo_match, traj_dist):
    """
    Ponderación sugerida:
    - Hausdorff (Forma general): 40%
    - IoU (Solapamiento): 30%
    - Topología (Estructura): 20%
    - Trayectoria (Fluidez): 10%
    """
    
    # 1. Score de Hausdorff 
    h_dist = geo_metrics['hausdorff']
    tolerance = 10.0
    adjusted_h = max(0, h_dist - tolerance)
    score_h = max(0, 100 - (adjusted_h * 1.8))
    
    # 2. Score IoU
    score_iou = min(100, geo_metrics['iou'] * 1.5)
    
    # 3. Score Topología 
    score_topo = 100 if topo_match else 30
    
    # 4. Score Trayectoria 
    score_traj = max(0, 100 - (traj_dist * 3))
    
    # Cálculo Final
    final = (score_h * 0.45) + (score_iou * 0.20) + (score_topo * 0.25) + (score_traj * 0.10)
    
    return round(final, 2)