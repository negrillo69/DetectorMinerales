import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class MineralModel:
    def __init__(self, model_path):
        # Carga el modelo CNN entrenado en lugar de un modelo de scikit-learn
        self.model = load_model(model_path)  # Carga el modelo .keras (mejor modelo guardado)

    def predict(self, image_path):
        # Procesar la imagen
        img = image.load_img(image_path, target_size=(150, 150))  # Asegúrate de que el tamaño coincida con el entrenamiento
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch
        img_array = img_array / 255.0  # Normalización como en el entrenamiento

        # Realizar la predicción
        prediction = self.model.predict(img_array)

        # Obtener el índice de la clase con mayor probabilidad
        predicted_class_index = np.argmax(prediction)

        # Suponiendo que tienes un diccionario de clases, podrías hacer algo como esto:
        class_labels = {0: 'mineral1', 1: 'mineral2', 2: 'mineral3', 3: 'mineral4'}  # Ejemplo, cambia esto por tus clases reales
        predicted_class_label = class_labels[predicted_class_index]

        return predicted_class_label

# Ejemplo de uso:
# model = MineralModel('best_model.keras')
# resultado = model.predict('ruta_a_imagen.jpg')
# print("El mineral predicho es:", resultado)

