"""
app.py
------
Servidor Flask — Integración Backend + Frontend.

Rutas disponibles:
  GET  /              → Página principal (interfaz web)
  POST /predict       → Clasifica texto y retorna categoría + probabilidades
  GET  /metrics       → Retorna métricas del entrenamiento (K-Folds results)
  GET  /health        → Estado del servidor y modelo

Uso:
    python app.py
"""

import os
import sys
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Asegurar imports desde raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.preprocessor import preprocess
from backend.model_io import load_model, load_training_results, model_exists

# ──────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Cargar modelo al iniciar el servidor
MODEL = None
TRAINING_RESULTS = None

def initialize_model():
    global MODEL, TRAINING_RESULTS
    if model_exists():
        try:
            MODEL = load_model()
            TRAINING_RESULTS = load_training_results()
            print("  [OK] Modelo cargado correctamente.")
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar el modelo: {e}")
            MODEL = None
    else:
        print("  [AVISO] Modelo no encontrado.")
        print("  Ejecuta: python -m backend.train_model")


# ──────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Sirve la página principal del portal de soporte."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de clasificación.

    Recibe JSON con:
        {
          "subject": "Título del ticket (opcional)",
          "description": "Texto de la solicitud"
        }

    Retorna JSON con:
        {
          "ticket_id": "TKT-XXXXXX",
          "category": "BILLING",
          "probabilities": { "BILLING": 0.85, "ORDER": 0.07, ... },
          "timestamp": "2026-04-15T10:30:00"
        }
    """
    if MODEL is None:
        return jsonify({
            'error': 'Modelo no disponible. Ejecuta primero: python -m backend.train_model'
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No se recibieron datos JSON.'}), 400

    subject     = data.get('subject', '').strip()
    description = data.get('description', '').strip()

    # Combinar subject + description para mayor contexto
    full_text = f"{subject} {description}".strip()

    if not full_text:
        return jsonify({'error': 'El texto de la solicitud no puede estar vacío.'}), 400

    # Preprocesar
    tokens = preprocess(full_text)

    if not tokens:
        return jsonify({'error': 'El texto no contiene palabras válidas después del preprocesamiento.'}), 400

    # Predicción con probabilidades
    predicted_class, probabilities = MODEL.predict_with_probabilities(tokens)

    # Ordenar probabilidades de mayor a menor
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

    # Generar ID de ticket único
    ticket_id = f"TKT-{str(uuid.uuid4()).upper()[:6]}"

    response = {
        'ticket_id':     ticket_id,
        'category':      predicted_class,
        'probabilities': sorted_probs,
        'timestamp':     datetime.now().isoformat(),
        'tokens_count':  len(tokens),
        'subject':       subject or 'Sin asunto',
        'description':   description[:200] + ('...' if len(description) > 200 else '')
    }

    return jsonify(response)


@app.route('/metrics')
def metrics():
    """
    Retorna las métricas del entrenamiento (K-Folds results) en JSON.
    Usadas por la interfaz web para mostrar el panel de estadísticas.
    """
    if TRAINING_RESULTS is None:
        return jsonify({'error': 'Resultados de entrenamiento no disponibles.'}), 404

    return jsonify(TRAINING_RESULTS)


@app.route('/health')
def health():
    """Estado del servidor."""
    return jsonify({
        'status':        'ok' if MODEL is not None else 'model_not_loaded',
        'model_loaded':  MODEL is not None,
        'classes':       sorted(MODEL.classes) if MODEL else [],
        'vocab_size':    MODEL.vocab_size if MODEL else 0,
        'timestamp':     datetime.now().isoformat()
    })


# ──────────────────────────────────────────────
# INICIAR SERVIDOR
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  PORTAL DE SOPORTE — CLASIFICADOR NAÏVE BAYES")
    print("  Universidad Rafael Landívar | IA 2026")
    print("="*55)
    initialize_model()
    print("\n  Servidor iniciando en: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)