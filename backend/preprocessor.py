"""
preprocessor.py
---------------
Módulo de preprocesamiento de texto para el clasificador Naïve Bayes.
Realiza: limpieza, tokenización, eliminación de stopwords y stemming.
"""

import re
import string
import nltk

# Descargar recursos necesarios de NLTK (solo la primera vez)
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Stopwords en inglés
STOPWORDS = set(stopwords.words('english'))

# Stemmer de Porter
stemmer = PorterStemmer()


def remove_placeholders(text: str) -> str:
    """
    Elimina placeholders del tipo {{Order Number}}, {{Customer Name}}, etc.
    Requerido por el dataset de Bitext.
    """
    return re.sub(r'\{\{[^}]+\}\}', '', text)


def clean_text(text: str) -> str:
    """
    Limpieza general del texto:
    - Convierte a minúsculas
    - Elimina placeholders
    - Elimina URLs
    - Elimina caracteres especiales y puntuación
    - Elimina dígitos
    - Elimina espacios extra
    """
    # Minúsculas
    text = text.lower()

    # Eliminar placeholders tipo {{...}}
    text = remove_placeholders(text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Eliminar caracteres especiales y puntuación (mantener letras y espacios)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Eliminar dígitos
    text = re.sub(r'\d+', '', text)

    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str) -> list:
    """
    Tokeniza el texto usando NLTK word_tokenize.
    Retorna lista de tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Elimina stopwords del inglés usando NLTK.
    También elimina tokens de longitud <= 1.
    """
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def apply_stemming(tokens: list) -> list:
    """
    Aplica stemming de Porter a cada token.
    Reduce palabras a su raíz (ej: 'ordering' → 'order').
    """
    return [stemmer.stem(token) for token in tokens]


def preprocess(text: str) -> list:
    """
    Pipeline completo de preprocesamiento:
    1. Limpieza del texto
    2. Tokenización
    3. Eliminación de stopwords
    4. Stemming

    Parámetros:
        text (str): Texto crudo de entrada.

    Retorna:
        list: Lista de tokens procesados.
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    return tokens


def preprocess_to_string(text: str) -> str:
    """
    Convierte el pipeline de preprocesamiento a un string unificado.
    Útil para construcción del vocabulario.
    """
    return ' '.join(preprocess(text))