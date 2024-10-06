from flask import Flask, request, render_template
from pymongo import MongoClient
import random
from models.model import MineralModel

app = Flask(__name__)

# Conexión a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Minerales']  # Cambia esto por el nombre de tu base de datos
minerals_collection = db['Coleccion1']  # Nombre de la colección de mineralessssss

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']

    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return 'No selected file', 400

    # Aquí puedes realizar validaciones adicionales si es necesario
    image_path = f'static/{file.filename}'
    file.save(image_path)
    model_path = 'models/best_model.keras'
    model = MineralModel(model_path)
    resultado = model.predict(image_path)
    print(resultado)

    # Obtener un mineral aleatorio de la base de datos
    minerals = minerals_collection.find_one({"nombre": resultado})  # Asegúrate de que existe esta colección
    if not minerals:
        return 'No hay minerales en la base de datos', 404
    print(minerals)
    

    return render_template('results.html', mineral=minerals)

if __name__ == '__main__':
    app.run(debug=True)
