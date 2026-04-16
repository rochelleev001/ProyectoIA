"""
model_io.py
-----------
Módulo para guardar y cargar el modelo entrenado.
Usa pickle para serialización del objeto NaiveBayesClassifier.
"""

import pickle
import os
import json


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'naive_bayes_model.pkl')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'training_results.json')


def save_model(model, filepath: str = MODEL_PATH):
    """
    Guarda el modelo entrenado en un archivo pickle.

    Parámetros:
        model (NaiveBayesClassifier): Modelo entrenado.
        filepath (str): Ruta del archivo de destino.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Modelo guardado en: {filepath} ({size_kb:.1f} KB)")


def load_model(filepath: str = MODEL_PATH):
    """
    Carga el modelo entrenado desde un archivo pickle.

    Parámetros:
        filepath (str): Ruta del archivo del modelo.

    Retorna:
        NaiveBayesClassifier: Modelo cargado.

    Lanza:
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Modelo no encontrado en '{filepath}'.\n"
            f"Ejecuta primero: python -m backend.train_model"
        )
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"  Modelo cargado desde: {filepath}")
    print(f"  Clases disponibles: {sorted(model.classes)}")
    return model

