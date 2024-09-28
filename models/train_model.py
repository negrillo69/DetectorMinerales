from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Crea la carpeta model si no existe
os.makedirs('models', exist_ok=True)

# Cargar el conjunto de datos (Ejemplo: Iris)
data = load_iris()
X, y = data.data, data.target

# Crear y entrenar el modelo
model = RandomForestClassifier()
model.fit(X, y)

# Guardar el modelo en un archivo .pkl
joblib.dump(model, 'models/modelo.pkl')  # Esto guarda el modelo en la carpeta models
