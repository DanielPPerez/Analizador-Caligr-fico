from app.core.config import HAUSDORFF_FACTOR, HAUSDORFF_TOLERANCE


def calculate_final_score(geo_metrics, topo_match, traj_dist, cosine_segment_score=50.0):
    """
    Ponderación orientada a tutor. Escalado para resolución 128×128.
    """
    score_proc = geo_metrics.get("procrustes_score", 0.0)

    h_dist = geo_metrics.get("hausdorff", 999.0)
    adjusted_h = max(0, h_dist - HAUSDORFF_TOLERANCE)
    score_h = max(0, 100 - (adjusted_h * HAUSDORFF_FACTOR))
    
    # 3. SSIM 
    score_ssim = geo_metrics.get("ssim_score", 0.0)
    
    # 4. Topología
    score_topo = 100 if topo_match else 30
    
    # 5. Trayectoria
    score_traj = max(0, 100 - (traj_dist * 3))
    
    # 6. Similitud de coseno por segmentos (ángulos)
    score_cos = max(0.0, min(100.0, cosine_segment_score))
    
    
    final = (
        score_proc * 0.28
        + score_h * 0.12
        + score_ssim * 0.20
        + score_topo * 0.25
        + score_traj * 0.10
        + score_cos * 0.05
    )
    
    return round(final, 2)