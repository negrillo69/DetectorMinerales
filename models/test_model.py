from model import MineralModel

def main():
    model_path = 'best_model.keras'  # Asegúrate de que este sea el camino correcto a tu modelo
    image_path = r'C:\Users\xexo_\OneDrive\Escritorio\U\images.jpg'  # Cambia esto a la ruta real de tu imagen

    # Inicializar el modelo
    model = MineralModel(model_path)

    # Realizar la predicción
    resultado = model.predict(image_path)
    
    if resultado:
        print("El mineral predicho es:", resultado)
    else:
        print("No se pudo realizar la predicción.")

if __name__ == '__main__':
    main()
