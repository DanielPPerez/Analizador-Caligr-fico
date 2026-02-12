# =============================================================================
# CLASIFICADOR RESNET (próxima implementación)
# =============================================================================
# Usar: from app.core.processor import get_image_for_resnet
#       img_128 = get_image_for_resnet(img_bytes)
#       char = model.predict(img_128)
# =============================================================================

# from torchvision.models import resnet18  # Descomentar cuando se implemente
# from app.core.processor import get_image_for_resnet


# def load_resnet_classifier(weights_path: str):
#     """Cargar modelo ResNet fine-tuned para clasificación de caracteres."""
#     pass  # model = resnet18(...); model.load_state_dict(...); return model


# def classify_char(model, img_bytes) -> str:
#     """Clasificar el carácter dibujado. Retorna 'A', 'b', '1', etc."""
#     # img = get_image_for_resnet(img_bytes)
#     # ... preprocess for model input
#     # return model(img)
#     pass
