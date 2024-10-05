import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class MineralModel:
    def __init__(self, model_path):
        # Carga el modelo CNN entrenado en lugar de un modelo de scikit-learn
        try:
            self.model = load_model(model_path)  # Carga el modelo .keras (mejor modelo guardado)
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            print("El modelo no está disponible.")
            return None

        try:
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
            class_labels = {
                0: 'Atacamita', 
                1: 'Bornita', 
                2: 'Brochantita',
                3: 'Calcopirita', 
                4: 'Calcosina', 
                5: 'Chalcantita',
                6: 'Cobre nativo', 
                7: 'Enargita', 
                8: 'Galena',
                9: 'Malaquita', 
                10: 'Pirita'
            }  # Cambia esto por tus clases reales
            
            predicted_class_label = class_labels[predicted_class_index]

            return predicted_class_label

        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return None

# Ejemplo de uso:
if __name__ == '__main__':
    model = MineralModel('best_model.keras')
    image_path = 'ruta_a_imagen.jpg'  # Cambia esto a la ruta real de tu imagen
    resultado = model.predict(image_path)
    if resultado:
        print("El mineral predicho es:", resultado)
    else:
        print("No se pudo realizar la predicción.")
