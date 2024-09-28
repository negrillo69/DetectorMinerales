# DetectorMinerales
# Proyecto de Reconocimiento de Minerales

Este proyecto utiliza un modelo de aprendizaje automático para identificar minerales a partir de imágenes. La aplicación está desarrollada en Python utilizando Flask para el backend, y MongoDB para la gestión de datos.

## Requisitos

- Python 3.x
- MongoDB
- Google Drive (para almacenar imágenes)

## Estructura del Proyecto

/ReconocimientoMinerales
│
│
├── models
│   ├── __init__.py                     # Inicializador del paquete
│   ├── model.py                        # Lógica de interacción con MongoDB y modelo de IA
│   └── train_model.py                  # Código para entrenar y guardar el modelo
│
├── templates
│   ├── upload.html                     # Formulario para subir imágenes
│   └── results.html                    # Página para mostrar los resultados
│
├── static
│   ├── styles.css                      # Estilos CSS para la aplicación
│   └── script.js                       # Código JavaScript (si es necesario)
│
├── train_model.py                      # Archivo que entrena el modelo
├── app.py                              # Archivo principal de la aplicación Flask
│
├── requirements.txt                    # Lista de dependencias del proyecto
├── Instrucciones_Virtualenv.ipynb      # Instrucciones Virtualenv
└── README.md                           # Este archivo