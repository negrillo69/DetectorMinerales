from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Ruta local de Google Drive sincronizado en tu sistema (esto deberia ser con links)
LOCAL_DRIVE_PATH = r'G:\Mi unidad\IA-imagenes'  # Ruta local de tu Google Drive

def entrenar_modelo():
    # Generador de imágenes desde directorios, con un 20% para validación
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Carga de imágenes directamente desde las carpetas locales en Google Drive
    train_generator = datagen.flow_from_directory(
        directory=LOCAL_DRIVE_PATH,
        target_size=(150, 150),   # Cambia el tamaño de las imágenes si es necesario
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

    # Crear el modelo CNN con Leaky ReLU y más filtros
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(150, 150, 3)),  # Aumenta a 64 filtros
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),  # Aumenta a 128 filtros
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3)),  # Aumenta a 256 filtros
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512),  # Capa densa con 512 neuronas
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')  # Número de clases
    ])

    # Compilar el modelo con un optimizador Adam y un learning rate ajustado
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Guardar el mejor modelo
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    # Entrenar el modelo
    model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[checkpoint])
    print("Entrenamiento completo. Modelo guardado.")

if __name__ == '__main__':
    entrenar_modelo()

