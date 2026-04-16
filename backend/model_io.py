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


def save_training_results(results: dict, filepath: str = RESULTS_PATH):
    """
    Guarda los resultados del entrenamiento (métricas K-Folds) en JSON.
    Convierte tipos no serializables para compatibilidad.

    Parámetros:
        results (dict): Resultados del K-Folds Cross Validation.
        filepath (str): Ruta del archivo JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convertir a tipos JSON-serializables
    def convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return str(obj)

    serializable_results = convert(results)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en: {filepath}")


def load_training_results(filepath: str = RESULTS_PATH) -> dict:
    """
    Carga los resultados del entrenamiento desde JSON.

    Parámetros:
        filepath (str): Ruta del archivo JSON.

    Retorna:
        dict: Resultados del entrenamiento.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def model_exists(filepath: str = MODEL_PATH) -> bool:
    """
    Verifica si el modelo entrenado existe.

    Retorna:
        bool: True si el modelo existe.
    """
    return os.path.exists(filepath)