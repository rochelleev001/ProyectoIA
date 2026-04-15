"""
naive_bayes.py
--------------
Implementación manual del algoritmo Naïve Bayes Multinomial para
clasificación de texto.

Técnicas implementadas:
  - Bag of Words (BoW): representación de documentos como vectores de frecuencia.
  - Laplace Smoothing: evita probabilidad cero para palabras no vistas.
  - Suma de logaritmos: evita underflow numérico al multiplicar probabilidades pequeñas.
"""

import math
from collections import defaultdict


class NaiveBayesClassifier:
    """
    Clasificador Naïve Bayes Multinomial implementado desde cero.

    Atributos:
        vocabulary (set): Vocabulario construido desde el corpus de entrenamiento.
        class_priors (dict): Probabilidades a priori log P(clase).
        word_likelihoods (dict): log P(palabra | clase) con Laplace Smoothing.
        classes (list): Lista de clases únicas.
        class_word_counts (dict): Conteo de palabras por clase.
        class_total_words (dict): Total de palabras por clase.
        vocab_size (int): Tamaño del vocabulario.
    """

    def __init__(self):
        self.vocabulary = set()
        self.class_priors = {}
        self.word_likelihoods = {}
        self.classes = []
        self.class_word_counts = {}
        self.class_total_words = {}
        self.vocab_size = 0

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------

    def build_vocabulary(self, documents: list) -> set:
        """
        Construye el vocabulario (Bag of Words) a partir del corpus.

        Parámetros:
            documents (list): Lista de listas de tokens (corpus preprocesado).

        Retorna:
            set: Vocabulario único del corpus.
        """
        vocab = set()
        for tokens in documents:
            for token in tokens:
                vocab.add(token)
        return vocab

    def fit(self, documents: list, labels: list):
        """
        Entrena el modelo Naïve Bayes Multinomial.

        Algoritmo:
            1. Construir el vocabulario desde el corpus.
            2. Calcular probabilidades a priori: log P(c) = log(N_c / N_total)
            3. Para cada clase, contar frecuencia de cada palabra.
            4. Calcular verosimilitud con Laplace Smoothing:
               log P(w|c) = log((count(w,c) + 1) / (total_words_c + |V|))

        Parámetros:
            documents (list): Lista de listas de tokens preprocesados.
            labels (list): Lista de etiquetas de clase correspondientes.
        """
        # Construir vocabulario
        self.vocabulary = self.build_vocabulary(documents)
        self.vocab_size = len(self.vocabulary)
        self.classes = list(set(labels))

        total_docs = len(documents)

        # Inicializar contadores
        class_doc_counts = defaultdict(int)
        self.class_word_counts = {cls: defaultdict(int) for cls in self.classes}
        self.class_total_words = defaultdict(int)

        # Contar documentos y palabras por clase
        for tokens, label in zip(documents, labels):
            class_doc_counts[label] += 1
            for token in tokens:
                self.class_word_counts[label][token] += 1
                self.class_total_words[label] += 1

        # -------------------------------------------------------
        # Calcular log P(clase) — Probabilidad a priori
        # log P(c) = log(N_c / N_total)
        # -------------------------------------------------------
        self.class_priors = {}
        for cls in self.classes:
            self.class_priors[cls] = math.log(class_doc_counts[cls] / total_docs)

        # -------------------------------------------------------
        # Calcular log P(palabra | clase) — Verosimilitud con Laplace Smoothing
        # P(w|c) = (count(w,c) + 1) / (total_words_c + |V|)
        # Usamos +1 en numerador y +|V| en denominador (Laplace α=1)
        # -------------------------------------------------------
        self.word_likelihoods = {}
        for cls in self.classes:
            self.word_likelihoods[cls] = {}
            denominator = self.class_total_words[cls] + self.vocab_size
            for word in self.vocabulary:
                count = self.class_word_counts[cls].get(word, 0)
                # log para evitar underflow numérico
                self.word_likelihoods[cls][word] = math.log((count + 1) / denominator)

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------

    def predict_single(self, tokens: list) -> tuple:
        """
        Predice la clase para un documento tokenizado.

        Usa la suma de logaritmos para calcular:
            log P(c|d) ∝ log P(c) + Σ log P(w|c)

        Palabras fuera del vocabulario se manejan con Laplace Smoothing
        usando el denominador correspondiente.

        Parámetros:
            tokens (list): Lista de tokens preprocesados del documento.

        Retorna:
            tuple: (clase_predicha, dict de scores por clase)
        """
        class_scores = {}

        for cls in self.classes:
            # Inicio con log prior
            score = self.class_priors[cls]
            denominator = self.class_total_words[cls] + self.vocab_size

            for token in tokens:
                if token in self.word_likelihoods[cls]:
                    # Palabra conocida: usar verosimilitud precomputada
                    score += self.word_likelihoods[cls][token]
                else:
                    # Palabra desconocida (OOV): Laplace con count=0
                    score += math.log(1 / denominator)

            class_scores[cls] = score

        # Clase con mayor score logarítmico
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class, class_scores

    def predict(self, documents: list) -> list:
        """
        Predice la clase para una lista de documentos tokenizados.

        Parámetros:
            documents (list): Lista de listas de tokens.

        Retorna:
            list: Lista de clases predichas.
        """
        return [self.predict_single(tokens)[0] for tokens in documents]

    def predict_with_probabilities(self, tokens: list) -> tuple:
        """
        Predice clase y retorna probabilidades normalizadas por clase.
        Convierte los log-scores a probabilidades aproximadas usando softmax.

        Parámetros:
            tokens (list): Lista de tokens preprocesados.

        Retorna:
            tuple: (clase_predicha, dict {clase: probabilidad})
        """
        predicted_class, class_scores = self.predict_single(tokens)

        # Softmax sobre log-scores para normalizar
        max_score = max(class_scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in class_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: round(exp_scores[cls] / total, 4) for cls in exp_scores}

        return predicted_class, probabilities

    # ------------------------------------------------------------------
    # INFORMACIÓN DEL MODELO
    # ------------------------------------------------------------------

    def get_top_words_per_class(self, n: int = 10) -> dict:
        """
        Retorna las N palabras más discriminativas por clase
        (las de mayor log-likelihood).

        Parámetros:
            n (int): Número de palabras top a retornar.

        Retorna:
            dict: {clase: [(palabra, log_prob), ...]}
        """
        top_words = {}
        for cls in self.classes:
            sorted_words = sorted(
                self.word_likelihoods[cls].items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_words[cls] = sorted_words[:n]
        return top_words