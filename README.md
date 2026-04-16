# HelpDesk AI — Clasificador Naïve Bayes

> **Proyecto de Inteligencia Artificial | Universidad Rafael Landívar | Primer Semestre 2026**
> Clasificación automática de solicitudes de soporte usando Naïve Bayes Multinomial implementado desde cero.

---

## Descripción

Sistema inteligente de clasificación de tickets de soporte al cliente. Implementa el algoritmo **Naïve Bayes Multinomial** completamente desde cero (sin scikit-learn ni equivalentes), utilizando:

- **Bag of Words** para representación de documentos
- **Laplace Smoothing** para palabras no vistas
- **Suma de logaritmos** para evitar underflow numérico
- **K-Folds Cross Validation** (K=5) implementado manualmente
- Evaluación completa: Precisión, Recall, F1-Score, Accuracy, Matriz de Confusión

El sistema incluye una interfaz web de tipo portal de soporte donde el usuario puede ingresar una solicitud y recibir la categoría predicha en tiempo real.

---

## Estructura del Proyecto

```
ProyectoIA/
│
├── backend/
│   ├── __init__.py
│   ├── preprocessor.py     # Limpieza, tokenización, stopwords, stemming
│   ├── naive_bayes.py      # Algoritmo Naïve Bayes desde cero
│   ├── trainer.py          # K-Folds + métricas implementadas manualmente
│   ├── model_io.py         # Guardar/cargar modelo (pickle + JSON)
│   └── train_model.py      # Script principal de entrenamiento
│
├── model/
│   ├── naive_bayes_model.pkl   # Modelo entrenado (generado al entrenar)
│   └── training_results.json  # Resultados K-Folds (generado al entrenar)
│
├── static/
│   ├── css/style.css       # Estilos de la interfaz web
│   └── js/app.js           # Lógica del frontend
│
├── templates/
│   └── index.html          # Página principal del portal
│
├── app.py                  # Servidor Flask (backend + frontend)
├── requirements.txt
└── README.md
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/rochelleev001/ProyectoIA.git
cd ProyectoIA
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso

### Paso 1: Entrenar el modelo

```bash
# Descarga automática del dataset de Bitext desde HuggingFace y entrena (original y recomendado)
python -m backend.train_model

# Opciones adicionales:
python -m backend.train_model --sample 5000   # Solo 5000 muestras (prueba rápida)
python -m backend.train_model --k 10          # K-Folds con K=10
python -m backend.train_model --csv data/mi_dataset.csv  # Dataset local CSV
```

El entrenamiento descargará automáticamente el dataset de Bitext (~26,872 instancias) y ejecutará K-Folds Cross Validation. Puede tardar entre 5 y 15 minutos en la primera ejecución.

**Salida esperada:**
```
[1/4] Cargando dataset...        → 26,872 instancias, 11 clases
[2/4] Preprocesando textos...    → Vocabulario: ~8,000 palabras
[3/4] K-Folds Cross Validation   → Accuracy: ~0.93 ± 0.002
[4/4] Guardando modelo...        → model/naive_bayes_model.pkl
```

### Paso 2: Iniciar el servidor web

```bash
python app.py
```

Abre el navegador en: **http://127.0.0.1:5000**

---

## Dataset

**Bitext Customer Support LLM Chatbot Training Dataset**
- URL: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- 26,872 solicitudes con etiquetas verificadas
- 11 categorías: ORDER, BILLING, SHIPPING, REFUND, ACCOUNT, CANCELLATION, CONTACT, DELIVERY, FEEDBACK, INVOICE, PAYMENT

---

## Pipeline de Preprocesamiento

1. **Limpieza**: Minúsculas, eliminación de placeholders `{{...}}`, URLs, caracteres especiales
2. **Tokenización**: `nltk.word_tokenize()`
3. **Stopwords**: Eliminación con lista NLTK en inglés
4. **Stemming**: Porter Stemmer (`nltk.stem.PorterStemmer`)

---

## Algoritmo Naïve Bayes

**Entrenamiento:**
```
P(c) = N_c / N_total                              (prior log)
P(w|c) = (count(w,c) + 1) / (total_words_c + |V|)  (Laplace Smoothing)
```

**Inferencia (suma de logaritmos):**
```
log P(c|d) ∝ log P(c) + Σ log P(w|c)
```

---

## Tecnologías Utilizadas

| Componente      | Tecnología                         |
|-----------------|------------------------------------|
| Backend         | Python 3.11                         |
| Clasificador    | Implementación manual (sin sklearn) |
| NLP básico      | NLTK (solo tokenización/stopwords) |
| Dataset         | HuggingFace `datasets`             |
| Persistencia    | Pickle + JSON                      |
| Servidor web    | Flask                              |
| Frontend        | HTML5 + CSS3 + JavaScript vanilla  |

---

## Endpoints de la API

| Método | Ruta        | Descripción                        |
|--------|-------------|------------------------------------|
| GET    | `/`         | Portal web principal               |
| POST   | `/predict`  | Clasificar texto → JSON            |
| GET    | `/metrics`  | Métricas de entrenamiento → JSON   |
| GET    | `/health`   | Estado del servidor y modelo       |

### Ejemplo `/predict`:
```json
POST /predict
{"subject": "Wrong charge", "description": "I was charged twice for my order"}

→ {
    "ticket_id": "TKT-A3F7B2",
    "category": "BILLING",
    "probabilities": {"BILLING": 0.87, "ORDER": 0.06, ...},
    "timestamp": "2026-04-15T10:30:00"
  }
```

---

