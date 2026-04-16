"""
trainer.py
----------
Módulo de entrenamiento y evaluación del clasificador Naïve Bayes.

Implementa desde cero:
  - K-Folds Cross Validation (K=5 por defecto)
  - Métricas: Precisión, Recall, F1-Score por clase
  - Accuracy global y Macro F1
  - Matriz de Confusión
"""

import random
import math
from collections import defaultdict
from backend.naive_bayes import NaiveBayesClassifier
from backend.preprocessor import preprocess


# ========================
# MÉTRICAS DESDE CERO
# ========================

def build_confusion_matrix(y_true: list, y_pred: list, classes: list) -> dict:
    """
    Construye la Matriz de Confusión como diccionario anidado.

    La matriz[real][predicho] = número de instancias.

    Parámetros:
        y_true (list): Etiquetas reales.
        y_pred (list): Etiquetas predichas.
        classes (list): Lista de clases únicas.

    Retorna:
        dict: Matriz de confusión {clase_real: {clase_pred: count}}
    """
    matrix = {cls: {c: 0 for c in classes} for cls in classes}
    for real, pred in zip(y_true, y_pred):
        if real in matrix and pred in matrix[real]:
            matrix[real][pred] += 1
    return matrix


def compute_metrics(y_true: list, y_pred: list, classes: list) -> dict:
    """
    Calcula métricas de evaluación por clase y globales desde cero.

    Para cada clase C:
      - TP (True Positives):  predicho=C y real=C
      - FP (False Positives): predicho=C y real≠C
      - FN (False Negatives): predicho≠C y real=C
      - TN (True Negatives):  predicho≠C y real≠C

    Fórmulas:
      Precisión  = TP / (TP + FP)   → de todas las predicciones de C, ¿cuántas son correctas?
      Recall     = TP / (TP + FN)   → de todos los reales de C, ¿cuántos se detectaron?
      F1-Score   = 2 * (P * R) / (P + R)
      Accuracy   = Σ TP / N_total
      Macro F1   = promedio de F1 por clase

    Parámetros:
        y_true (list): Etiquetas reales.
        y_pred (list): Etiquetas predichas.
        classes (list): Lista de clases.

    Retorna:
        dict: Métricas completas.
    """
    metrics = {}
    total = len(y_true)
    correct = sum(1 for r, p in zip(y_true, y_pred) if r == p)

    accuracy = correct / total if total > 0 else 0.0

    f1_scores = []
    for cls in classes:
        tp = sum(1 for r, p in zip(y_true, y_pred) if r == cls and p == cls)
        fp = sum(1 for r, p in zip(y_true, y_pred) if r != cls and p == cls)
        fn = sum(1 for r, p in zip(y_true, y_pred) if r == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1_score':  round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        'per_class': metrics,
        'accuracy':  round(accuracy, 4),
        'macro_f1':  round(macro_f1, 4)
    }


# ================================
# K-FOLDS CROSS VALIDATION MANUAL
# ================================

def create_folds(documents: list, labels: list, k: int = 5, seed: int = 42) -> list:
    """
    Divide el dataset en K particiones (folds) de forma estratificada
    para mantener distribución de clases proporcional en cada fold.

    Parámetros:
        documents (list): Lista de documentos tokenizados.
        labels (list): Lista de etiquetas.
        k (int): Número de folds (mínimo 5 según requerimientos).
        seed (int): Semilla para reproducibilidad.

    Retorna:
        list: Lista de K folds, cada uno con índices de ese fold.
    """
    random.seed(seed)

    # agrupar índices por clase para estratificación
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # mezclar índices dentro de cada clase
    for cls in class_indices:
        random.shuffle(class_indices[cls])

    # distribuir en K folds
    folds = [[] for _ in range(k)]
    for cls, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    # mezclar cada fold
    for fold in folds:
        random.shuffle(fold)

    return folds


def k_folds_cross_validation(documents: list, labels: list, k: int = 5) -> dict:
    """
    Ejecuta K-Folds Cross Validation completo.

    En cada iteración:
      - 1 fold → conjunto de prueba
      - K-1 folds → conjunto de entrenamiento
      - Entrena el modelo y evalúa métricas

    Promedia las métricas de todos los folds y calcula varianza.

    Parámetros:
        documents (list): Lista de listas de tokens preprocesados.
        labels (list): Lista de etiquetas.
        k (int): Número de folds.

    Retorna:
        dict: Resultados completos con métricas promedio, varianza y por fold.
    """
    print(f"\n{'='*60}")
    print(f"  INICIANDO K-FOLDS CROSS VALIDATION (K={k})")
    print(f"{'='*60}")

    classes = sorted(list(set(labels)))
    folds = create_folds(documents, labels, k)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_idx in range(k):
        print(f"\n  [Fold {fold_idx + 1}/{k}] Entrenando...")

        # índices de prueba: fold actual
        test_indices = set(folds[fold_idx])

        # índices de entrenamiento: todos los demás folds
        train_indices = []
        for j in range(k):
            if j != fold_idx:
                train_indices.extend(folds[j])

        # preparar datos de entrenamiento y prueba
        train_docs   = [documents[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_docs    = [documents[i] for i in test_indices]
        test_labels  = [labels[i] for i in test_indices]

        # entrenar modelo
        model = NaiveBayesClassifier()
        model.fit(train_docs, train_labels)

        # predecir
        y_pred = model.predict(test_docs)

        # métricas de este fold
        metrics = compute_metrics(test_labels, y_pred, classes)
        confusion = build_confusion_matrix(test_labels, y_pred, classes)

        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'per_class': metrics['per_class'],
            'confusion_matrix': confusion,
            'test_size': len(test_labels),
            'train_size': len(train_labels)
        }

        fold_results.append(fold_result)
        all_y_true.extend(test_labels)
        all_y_pred.extend(y_pred)

        print(f"     Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")

    # -------------------------------
    # promediar métricas entre folds
    # -------------------------------
    avg_accuracy = sum(r['accuracy'] for r in fold_results) / k
    avg_macro_f1 = sum(r['macro_f1'] for r in fold_results) / k

    # varianza de accuracy
    var_accuracy = sum((r['accuracy'] - avg_accuracy) ** 2 for r in fold_results) / k
    std_accuracy = math.sqrt(var_accuracy)

    # promediar métricas por clase
    avg_per_class = {}
    for cls in classes:
        avg_per_class[cls] = {
            'precision': round(sum(r['per_class'].get(cls, {}).get('precision', 0) for r in fold_results) / k, 4),
            'recall':    round(sum(r['per_class'].get(cls, {}).get('recall', 0) for r in fold_results) / k, 4),
            'f1_score':  round(sum(r['per_class'].get(cls, {}).get('f1_score', 0) for r in fold_results) / k, 4),
        }

    # matriz de confusión global sobre todas las predicciones
    global_confusion = build_confusion_matrix(all_y_true, all_y_pred, classes)
    global_metrics = compute_metrics(all_y_true, all_y_pred, classes)

    print(f"\n{'='*60}")
    print(f"  RESULTADOS FINALES K-FOLDS")
    print(f"  Accuracy promedio : {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Macro F1 promedio  : {avg_macro_f1:.4f}")
    print(f"{'='*60}\n")

    return {
        'k': k,
        'fold_results': fold_results,
        'avg_accuracy': round(avg_accuracy, 4),
        'avg_macro_f1': round(avg_macro_f1, 4),
        'std_accuracy': round(std_accuracy, 4),
        'avg_per_class': avg_per_class,
        'global_confusion': global_confusion,
        'global_metrics': global_metrics,
        'classes': classes
    }


# ===============================================
# ENTRENAR MODELO FINAL (sobre todo el dataset)
# ===============================================

def train_final_model(documents: list, labels: list) -> NaiveBayesClassifier:
    """
    Entrena el modelo final sobre el 100% del dataset.
    Este es el modelo que se guardará y se usará para predicciones en producción.

    Parámetros:
        documents (list): Lista de listas de tokens.
        labels (list): Lista de etiquetas.

    Retorna:
        NaiveBayesClassifier: Modelo entrenado.
    """
    print("\n  Entrenando modelo final sobre el dataset completo...")
    model = NaiveBayesClassifier()
    model.fit(documents, labels)
    print(f"  Modelo final entrenado.")
    print(f"  Vocabulario: {model.vocab_size:,} palabras")
    print(f"  Clases: {sorted(model.classes)}")
    return model