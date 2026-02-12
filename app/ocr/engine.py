# =============================================================================
# MOTOR OCR (próxima implementación)
# =============================================================================
# Integrar con clasificador ResNet o Tesseract según necesidad.
# El preprocesamiento en processor.py ya entrega imágenes 128x128 listas.
# =============================================================================

# from app.core.processor import get_image_for_resnet, preprocess_robust
# from app.models.classifier import classify_char  # cuando exista


# def recognize_char(img_bytes) -> str:
#     """Reconocer carácter desde imagen de celular."""
#     pass


# def recognize_word(img_bytes) -> str:
#     """Reconocer palabra completa (múltiples caracteres)."""
#     pass
