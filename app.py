from flask import Flask, request, render_template
from pymongo import MongoClient
import random
import models.model

app = Flask(__name__)

# Conexión a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['minerales_db']  # Cambia esto por el nombre de tu base de datos

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

    # Obtener un mineral aleatorio de la base de datos
    minerals = list(db.minerales.find())  # Asegúrate de que existe esta colección
    random_mineral = random.choice(minerals)

    return render_template('results.html', mineral=random_mineral)

if __name__ == '__main__':
    app.run(debug=True)
