"""
train_model.py
--------------
Script principal de entrenamiento del clasificador.

Pasos:
  1. Carga el dataset (Bitext Customer Support de HuggingFace o CSV local)
  2. Preprocesa todos los textos
  3. Ejecuta K-Folds Cross Validation (K=5)
  4. Entrena el modelo final sobre el dataset completo
  5. Guarda el modelo y los resultados

Uso:
    python -m backend.train_model

    O con opciones:
    python -m backend.train_model --sample 5000
    python -m backend.train_model --csv data/dataset.csv --k 10
"""

import os
import sys
import argparse
import time

# asegurar que el directorio raíz esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocessor import preprocess
from backend.trainer import k_folds_cross_validation, train_final_model
from backend.model_io import save_model, save_training_results


def load_dataset_from_huggingface(sample_size: int = None):
    """
    Carga el dataset de Bitext desde HuggingFace.
    Requiere: pip install datasets

    Parámetros:
        sample_size (int): Si se especifica, toma solo N muestras (para pruebas rápidas).

    Retorna:
        tuple: (textos, etiquetas)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: Instala 'datasets' con: pip install datasets")
        sys.exit(1)

    print("  Descargando dataset de Bitext desde HuggingFace...")
    print("  (Esto puede tardar unos minutos la primera vez)")

    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )

    texts  = [row['instruction'] for row in dataset]
    labels = [row['category'] for row in dataset]

    if sample_size and sample_size < len(texts):
        import random
        random.seed(42)
        indices = random.sample(range(len(texts)), sample_size)
        texts  = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"  Usando muestra aleatoria de {sample_size:,} instancias.")

    return texts, labels


def load_dataset_from_csv(filepath: str):
    """
    Carga el dataset desde un archivo CSV local.
    Columnas esperadas: 'instruction' o 'text', y 'category' o 'label'.

    Parámetros:
        filepath (str): Ruta al archivo CSV.

    Retorna:
        tuple: (textos, etiquetas)
    """
    import csv

    texts, labels = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # detectar columna de texto
            text_col = next(
                (c for c in ['instruction', 'text', 'description', 'query'] if c in row), None
            )
            # detectar columna de etiqueta
            label_col = next(
                (c for c in ['category', 'label', 'intent', 'class'] if c in row), None
            )
            if text_col and label_col:
                texts.append(row[text_col])
                labels.append(row[label_col].upper())

    print(f"  Dataset cargado desde CSV: {len(texts):,} instancias.")
    return texts, labels

