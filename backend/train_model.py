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


def print_dataset_stats(labels: list, classes: list):
    """Imprime estadísticas del dataset."""
    print(f"\n  {'─'*50}")
    print(f"  ESTADÍSTICAS DEL DATASET")
    print(f"  {'─'*50}")
    print(f"  Total de instancias : {len(labels):,}")
    print(f"  Número de clases    : {len(classes)}")
    print(f"\n  Distribución por clase:")
    from collections import Counter
    counts = Counter(labels)
    for cls in sorted(classes):
        pct = counts[cls] / len(labels) * 100
        bar = '█' * int(pct / 2)
        print(f"    {cls:<20} {counts[cls]:>5,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Entrenar clasificador Naïve Bayes')
    parser.add_argument('--csv',    type=str, default=None, help='Ruta a CSV local (opcional)')
    parser.add_argument('--sample', type=int, default=None, help='Número de muestras (para pruebas rápidas)')
    parser.add_argument('--k',      type=int, default=5,    help='Número de folds (default: 5)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CLASIFICADOR NAÏVE BAYES — MESA DE AYUDA")
    print("  Universidad Rafael Landívar | IA 2026")
    print("="*60)

    # --------------------
    # 1. Cargar dataset
    # --------------------
    print("\n[1/4] Cargando dataset...")
    t0 = time.time()

    if args.csv:
        texts, labels = load_dataset_from_csv(args.csv)
    else:
        texts, labels = load_dataset_from_huggingface(sample_size=args.sample)

    classes = sorted(list(set(labels)))
    print(f"  Dataset cargado en {time.time() - t0:.1f}s")
    print_dataset_stats(labels, classes)

    # --------------------
    # 2. Preprocesamiento
    # -------------------
    print("[2/4] Preprocesando textos...")
    t1 = time.time()

    processed_docs = []
    total = len(texts)
    for i, text in enumerate(texts):
        tokens = preprocess(text)
        processed_docs.append(tokens)
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  Procesados: {i+1:,}/{total:,} ({pct:.1f}%)", end='\r')

    print(f"\n  Preprocesamiento completado en {time.time() - t1:.1f}s")

    # estadísticas de tokens
    all_tokens = [t for doc in processed_docs for t in doc]
    vocab = set(all_tokens)
    avg_len = sum(len(doc) for doc in processed_docs) / len(processed_docs)
    print(f"  Vocabulario único  : {len(vocab):,} palabras")
    print(f"  Promedio tokens/doc: {avg_len:.1f}")

    # -----------------------------
    # 3. K-folds cross validation
    # -----------------------------
    print(f"\n[3/4] Evaluando con K-Folds Cross Validation (K={args.k})...")
    t2 = time.time()

    cv_results = k_folds_cross_validation(processed_docs, labels, k=args.k)

    print(f"\n  Métricas por clase (promedio de {args.k} folds):")
    print(f"  {'Clase':<25} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'─'*55}")
    for cls, m in cv_results['avg_per_class'].items():
        print(f"  {cls:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
    print(f"  {'─'*55}")
    print(f"  {'Accuracy promedio':<25} {cv_results['avg_accuracy']:>10.4f}")
    print(f"  {'Macro F1 promedio':<25} {cv_results['avg_macro_f1']:>10.4f}")
    print(f"  {'Desv. estándar Acc':<25} ±{cv_results['std_accuracy']:>9.4f}")
    print(f"\n  K-Folds completado en {time.time() - t2:.1f}s")

    # ------------------------------------
    # 4. Entrenar modelo final y guardar 
    # ------------------------------------
    print("\n[4/4] Entrenando modelo final y guardando...")
    t3 = time.time()

    final_model = train_final_model(processed_docs, labels)

    save_model(final_model)
    save_training_results(cv_results)

    print(f"\n  Proceso completo en {time.time() - t0:.1f}s total")
    print("\n  ✓ Modelo listo para usar.")
    print("  ✓ Ejecuta 'python app.py' para iniciar el servidor web.\n")


if __name__ == '__main__':
    main()