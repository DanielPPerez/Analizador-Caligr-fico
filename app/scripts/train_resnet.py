import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
# Importamos la arquitectura que definimos antes
from app.ocr.resnet_model import build_resnet_char

def load_and_prep_emnist():
    """Carga EMNIST y lo adapta para ResNet 32x32"""
    import tensorflow_datasets as tfds
    print("Descargando/Cargando EMNIST ByClass...")
    ds_train, ds_info = tfds.load('emnist/byclass', split='train', as_supervised=True, with_info=True)
    ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)

    def preprocess(image, label):
        # EMNIST viene rotado y transpuesto por defecto, corregimos:
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.resize(image, (32, 32))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = ds_train.map(preprocess).cache().shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, ds_info.features['label'].num_classes

def train():
    # 1. Preparar datos
    train_ds, test_ds, num_classes = load_and_prep_emnist()
    
    # Nota: Para la Ñ/ñ, deberías concatenar aquí imágenes propias 
    # y subir num_classes a 64.
    
    # 2. Construir Modelo
    model = build_resnet_char(input_shape=(32, 32, 1), num_classes=num_classes)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Data Augmentation (Crucial para fotos de cuadernos)
    # Definimos transformaciones que simulan fotos reales
    datagen = ImageDataGenerator(
        rotation_range=15,      # Simula cámara no alineada
        width_shift_range=0.1,  # Simula mal centrado
        height_shift_range=0.1,
        zoom_range=0.2,         # Simula distancias variadas
        shear_range=0.1,        # Simula perspectiva
        brightness_range=[0.7, 1.3] # Simula sombras/poca luz
    )

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # 5. Entrenamiento
    print("Iniciando entrenamiento de ResNet...")
    model.fit(
        train_ds,
        epochs=30,
        validation_data=test_ds,
        callbacks=callbacks
    )

    # 6. Guardar Modelo
    model.save("app/models/resnet_trained.h5")
    print("Modelo guardado como resnet_trained.h5")

if __name__ == "__main__":
    train()