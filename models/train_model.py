from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# Ruta local de Google Drive sincronizado en tu sistema
LOCAL_DRIVE_PATH = r'G:\Mi unidad\IA-imagenes'

def entrenar_modelo():
    # Generador de imágenes desde directorios con aumento de datos
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # Dividir en 80% entrenamiento y 20% validación
        rotation_range=40,      # Rango de rotación para aumentar datos
        width_shift_range=0.2,  # Desplazamiento horizontal
        height_shift_range=0.2, # Desplazamiento vertical
        shear_range=0.2,        # Aplicar corte en el eje
        zoom_range=0.2,         # Aumento con zoom
        horizontal_flip=True,   # Voltear imágenes horizontalmente
        fill_mode='nearest'     # Llenar píxeles vacíos
    )

    # Carga de imágenes directamente desde las carpetas locales en Google Drive
    train_generator = datagen.flow_from_directory(
        directory=LOCAL_DRIVE_PATH,
        target_size=(150, 150),   
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        directory=LOCAL_DRIVE_PATH,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Crear el modelo CNN
    model = Sequential([
        Input(shape=(150, 150, 3)),  # Usar Input como primera capa
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),  # Dropout para reducir sobreajuste

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),  # Dropout adicional

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),  # Mantener ReLU aquí
        Dropout(0.5),  # Dropout más fuerte en la capa completamente conectada
        Dense(len(train_generator.class_indices), activation='softmax')  # Número de clases
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Guardar el mejor modelo
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    # Entrenar el modelo
    model.fit(train_generator, validation_data=validation_generator, epochs=30, callbacks=[checkpoint])  # Aumentar el número de épocas
    print("Entrenamiento completo. Modelo guardado.")

if __name__ == '__main__':
    entrenar_modelo()
