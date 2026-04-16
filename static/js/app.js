/**
 * app.js
 * ------
 * Lógica del frontend para el Portal de Soporte HelpDesk AI.
 *
 * Funcionalidades:
 *  - Envío de tickets al backend Flask (/predict)
 *  - Visualización de categoría predicha y barras de confianza
 *  - Carga y visualización de métricas K-Folds (/metrics)
 *  - Matriz de confusión dinámica con heatmap de colores
 *  - Ejemplos rápidos para pruebas
 */

'use strict';

// ══════════════════════════════════════════════
// DESCRIPCIONES DE CATEGORÍAS
// ══════════════════════════════════════════════

const CATEGORY_INFO = {
  ORDER:        { label: 'Gestión de Órdenes',       icon: '📦', desc: 'Consultas sobre pedidos, seguimiento y estado de órdenes.' },
  BILLING:      { label: 'Facturación',               icon: '💳', desc: 'Problemas con cargos, facturas y métodos de pago.' },
  SHIPPING:     { label: 'Envíos y Logística',        icon: '🚚', desc: 'Tiempos de entrega, paquetes perdidos o dañados.' },
  REFUND:       { label: 'Reembolsos',                icon: '↩️', desc: 'Solicitudes de devolución de dinero o crédito.' },
  ACCOUNT:      { label: 'Gestión de Cuenta',         icon: '👤', desc: 'Acceso, contraseñas, perfil y configuración.' },
  CANCELLATION: { label: 'Cancelaciones',             icon: '🚫', desc: 'Cancelación de pedidos o suscripciones.' },
  CONTACT:      { label: 'Información de Contacto',   icon: '📞', desc: 'Solicitudes de datos de contacto o atención.' },
  DELIVERY:     { label: 'Entrega',                   icon: '📬', desc: 'Problemas con la entrega del pedido.' },
  FEEDBACK:     { label: 'Retroalimentación',         icon: '⭐', desc: 'Comentarios, sugerencias y opiniones.' },
  INVOICE:      { label: 'Facturas',                  icon: '🧾', desc: 'Solicitudes o problemas con documentos de factura.' },
  PAYMENT:      { label: 'Pagos',                     icon: '💰', desc: 'Problemas al procesar pagos o transacciones.' },
  NEWSLETTER:   { label: 'Boletín',                   icon: '📧', desc: 'Suscripción o cancelación de boletines informativos.' },
  PROMO:        { label: 'Promociones',               icon: '🎁', desc: 'Cupones, descuentos y ofertas.' },
  REVIEW:       { label: 'Reseñas',                   icon: '✍️', desc: 'Opiniones sobre productos o servicios.' },
  SUPPORT:      { label: 'Soporte General',           icon: '🛠️', desc: 'Ayuda técnica general o consultas diversas.' },
};

function getCategoryInfo(category) {
  const key = (category || '').toUpperCase();
  return CATEGORY_INFO[key] || { label: category, icon: '📋', desc: 'Solicitud clasificada automáticamente.' };
}

// ══════════════════════════════════════════════
// EJEMPLOS RÁPIDOS
// ══════════════════════════════════════════════

const EXAMPLES = {
  order:    { subject: 'Estado de mi pedido', description: "I placed an order three days ago but I still haven't received any confirmation email. Could you please check the status of my order and let me know if everything is fine?" },
  billing:  { subject: 'Cargo incorrecto en mi tarjeta', description: "I noticed an incorrect charge on my credit card statement from last month. The amount doesn't match what I expected to pay. I need you to review my billing history and clarify this charge." },
  shipping: { subject: 'Paquete perdido en tránsito', description: "My package was supposed to arrive last week but the tracking shows it has been stuck in transit for days. I am worried it might be lost. Can you investigate what happened with my shipment?" },
  refund:   { subject: 'Solicitud de reembolso', description: "I would like to request a full refund for my recent purchase. The product did not match the description on the website and I am very disappointed. Please process my refund as soon as possible." },
  account:  { subject: 'No puedo acceder a mi cuenta', description: "I am unable to log into my account. I have tried resetting my password multiple times but I keep getting an error message. I need immediate help to recover access to my account." },
};

function fillExample(type) {
  const ex = EXAMPLES[type];
  if (!ex) return;
  document.getElementById('subject').value = ex.subject;
  document.getElementById('description').value = ex.description;
  updateCharCount();
  // Highlight briefly
  const form = document.querySelector('.ticket-form');
  form.style.transition = 'background 0.3s';
  form.style.background = 'rgba(61,139,255,0.04)';
  setTimeout(() => { form.style.background = ''; }, 400);
}

// ══════════════════════════════════════════════
// CONTADOR DE CARACTERES
// ══════════════════════════════════════════════

function updateCharCount() {
  const desc = document.getElementById('description');
  const counter = document.getElementById('charCount');
  if (desc && counter) counter.textContent = desc.value.length;
}

document.addEventListener('DOMContentLoaded', () => {
  const desc = document.getElementById('description');
  if (desc) desc.addEventListener('input', updateCharCount);
  checkServerHealth();
});

// ══════════════════════════════════════════════
// HEALTH CHECK
// ══════════════════════════════════════════════

async function checkServerHealth() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.model_loaded) {
      dot.classList.add('online');
      text.textContent = `Modelo listo · ${data.vocab_size.toLocaleString()} palabras`;
    } else {
      dot.classList.add('offline');
      text.textContent = 'Modelo no cargado';
    }
  } catch {
    dot.classList.add('offline');
    text.textContent = 'Sin conexión';
  }
}

// ══════════════════════════════════════════════
// NAVEGACIÓN ENTRE VISTAS
// ══════════════════════════════════════════════

function showView(name) {
  document.querySelectorAll('.view').forEach(v => {
    v.classList.remove('active');
    v.classList.add('hidden');
  });
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

  const view = document.getElementById(`view-${name}`);
  if (view) {
    view.classList.remove('hidden');
    view.classList.add('active');
  }

  const btns = document.querySelectorAll('.nav-btn');
  btns.forEach(b => {
    if (b.getAttribute('onclick')?.includes(name)) b.classList.add('active');
  });

  if (name === 'metrics') loadMetrics();
}

// ══════════════════════════════════════════════
// ENVÍO DE TICKET
// ══════════════════════════════════════════════

async function submitTicket() {
  const subjectEl     = document.getElementById('subject');
  const descEl        = document.getElementById('description');
  const submitBtn     = document.getElementById('submitBtn');

  const subject     = subjectEl.value.trim();
  const description = descEl.value.trim();

  if (!description) {
    shake(descEl);
    descEl.focus();
    return;
  }

  // Estado de carga
  setSubmitLoading(true);
  hideAllResults();

  try {
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ subject, description })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || 'Error desconocido al clasificar.');
      return;
    }

    showResult(data);

  } catch (err) {
    showError('No se pudo conectar con el servidor. ¿Está ejecutándose Flask?');
  } finally {
    setSubmitLoading(false);
  }
}

function retrySubmit() {
  hideAllResults();
  document.getElementById('resultEmpty').classList.remove('hidden');
}

function resetForm() {
  document.getElementById('subject').value      = '';
  document.getElementById('description').value  = '';
  updateCharCount();
  hideAllResults();
  document.getElementById('resultEmpty').classList.remove('hidden');
}

// ──────────────────────────────────────────────
// MOSTRAR RESULTADO
// ──────────────────────────────────────────────

function showResult(data) {
  const info = getCategoryInfo(data.category);

  // Ticket ID y timestamp
  document.getElementById('ticketIdDisplay').textContent = data.ticket_id || '—';
  document.getElementById('ticketTimestamp').textContent = formatTimestamp(data.timestamp);

  // Resumen del ticket (subject + description)
  document.getElementById('summarySubject').textContent =
    data.subject && data.subject !== 'Sin asunto' ? data.subject : '(Sin asunto)';
  document.getElementById('summaryDescription').textContent = data.description || '—';

  // Categoría principal
  document.getElementById('predictedCategory').textContent  = `${info.icon} ${data.category}`;
  document.getElementById('categoryDescription').textContent = info.desc;

  // Departamento enrutado
  document.getElementById('metaDepartment').textContent = info.label;

  // Tokens procesados
  document.getElementById('tokensCount').textContent = data.tokens_count ?? '—';

  // Barras de confianza
  renderConfidenceBars(data.probabilities, data.category);

  // Mostrar panel
  document.getElementById('resultEmpty').classList.add('hidden');
  document.getElementById('resultContent').classList.remove('hidden');
}

function renderConfidenceBars(probabilities, topCategory) {
  const container = document.getElementById('confidenceBars');
  container.innerHTML = '';

  // Ordenar de mayor a menor
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, prob]) => {
    const pct    = (prob * 100).toFixed(1);
    const isTop  = cls === topCategory;
    const info   = getCategoryInfo(cls);

    const row = document.createElement('div');
    row.className = 'conf-bar-row';
    row.innerHTML = `
      <span class="conf-bar-label">${info.icon} ${cls}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill ${isTop ? 'top' : pct < 5 ? 'low' : ''}"
             style="width: 0%"
             data-target="${pct}%"></div>
      </div>
      <span class="conf-bar-pct">${pct}%</span>
    `;
    container.appendChild(row);
  });

  // Animar barras con un pequeño delay
  requestAnimationFrame(() => {
    document.querySelectorAll('.conf-bar-fill[data-target]').forEach(bar => {
      setTimeout(() => {
        bar.style.width = bar.getAttribute('data-target');
      }, 50);
    });
  });
}

// ──────────────────────────────────────────────
// ESTADOS UI
// ──────────────────────────────────────────────

function hideAllResults() {
  ['resultEmpty', 'resultContent', 'resultError'].forEach(id => {
    document.getElementById(id)?.classList.add('hidden');
  });
}

function showError(msg) {
  document.getElementById('errorMsg').textContent = msg;
  document.getElementById('resultError').classList.remove('hidden');
}

function setSubmitLoading(loading) {
  const btn  = document.getElementById('submitBtn');
  const text = btn.querySelector('.btn-text');
  const icon = btn.querySelector('.btn-icon');
  if (loading) {
    btn.disabled = true;
    btn.classList.add('loading');
    text.textContent = 'Clasificando...';
    icon.textContent = '⟳';
  } else {
    btn.disabled = false;
    btn.classList.remove('loading');
    text.textContent = 'Clasificar Solicitud';
    icon.textContent = '→';
  }
}

function shake(el) {
  el.style.animation = 'none';
  el.offsetHeight; // reflow
  el.style.animation = 'shake 0.4s ease';
  el.addEventListener('animationend', () => { el.style.animation = ''; }, { once: true });
  // Agregar keyframe dinámicamente si no existe
  if (!document.getElementById('shakeStyle')) {
    const style = document.createElement('style');
    style.id = 'shakeStyle';
    style.textContent = `@keyframes shake { 0%,100%{transform:translateX(0)} 20%{transform:translateX(-6px)} 60%{transform:translateX(6px)} 80%{transform:translateX(-3px)} }`;
    document.head.appendChild(style);
  }
}

// ══════════════════════════════════════════════
// MÉTRICAS
// ══════════════════════════════════════════════

let metricsLoaded = false;

async function loadMetrics() {
  if (metricsLoaded) return;

  const loading = document.getElementById('metricsLoading');
  const content = document.getElementById('metricsContent');
  loading.style.display = 'flex';
  content.classList.add('hidden');

  try {
    const res  = await fetch('/metrics');
    if (!res.ok) throw new Error('Métricas no disponibles');
    const data = await res.json();

    renderGlobalCards(data);
    renderClassMetricsTable(data);
    renderFoldsGrid(data);
    renderConfusionMatrix(data);

    loading.style.display = 'none';
    content.classList.remove('hidden');
    metricsLoaded = true;

  } catch (err) {
    loading.innerHTML = `<p style="color:var(--accent-error)">⚠️ ${err.message}.<br>Asegúrate de haber entrenado el modelo primero.</p>`;
  }
}

// ── Cards globales ──

function renderGlobalCards(data) {
  const container = document.getElementById('globalCards');
  const cards = [
    { val: pct(data.avg_accuracy),  label: 'Accuracy Promedio' },
    { val: pct(data.avg_macro_f1),  label: 'Macro F1 Promedio' },
    { val: `±${pct(data.std_accuracy)}`, label: 'Desv. Estándar Acc' },
    { val: data.k,                   label: 'Número de Folds' },
    { val: data.classes?.length ?? '—', label: 'Clases' },
  ];
  container.innerHTML = cards.map(c => `
    <div class="global-card">
      <span class="card-val">${c.val}</span>
      <span class="card-label">${c.label}</span>
    </div>
  `).join('');
}

// ── Tabla de métricas por clase ──

function renderClassMetricsTable(data) {
  const tbody = document.getElementById('classMetricsBody');
  const perClass = data.avg_per_class || {};

  tbody.innerHTML = Object.entries(perClass).sort((a,b)=> b[1].f1_score - a[1].f1_score).map(([cls, m]) => `
    <tr>
      <td class="class-name-cell">${getCategoryInfo(cls).icon} ${cls}</td>
      <td>${metricCell(m.precision)}</td>
      <td>${metricCell(m.recall)}</td>
      <td>${metricCell(m.f1_score)}</td>
    </tr>
  `).join('');
}

function metricCell(val) {
  const pct = (val * 100).toFixed(1);
  return `
    <div class="metric-bar-cell">
      <span>${pct}%</span>
      <div class="mini-bar-track">
        <div class="mini-bar-fill" style="width:${pct}%"></div>
      </div>
    </div>
  `;
}

// ── Folds grid ──

function renderFoldsGrid(data) {
  const container = document.getElementById('foldsGrid');
  const folds = data.fold_results || [];
  container.innerHTML = folds.map(f => `
    <div class="fold-card">
      <div class="fold-num">Fold ${f.fold}</div>
      <div class="fold-acc">${pct(f.accuracy)}</div>
      <div class="fold-f1">Macro F1: ${pct(f.macro_f1)}</div>
    </div>
  `).join('');
}

// ── Matriz de confusión ──

function renderConfusionMatrix(data) {
  const container = document.getElementById('confusionMatrix');
  const matrix    = data.global_confusion || {};
  const classes   = data.classes || Object.keys(matrix);

  if (!classes.length) {
    container.innerHTML = '<p style="color:var(--text-muted)">No hay datos de la matriz de confusión.</p>';
    return;
  }

  // Encontrar valor máximo para normalizar colores
  let maxVal = 0;
  classes.forEach(r => classes.forEach(c => {
    const v = (matrix[r] || {})[c] || 0;
    if (v > maxVal) maxVal = v;
  }));

  // Encabezado de columnas
  let html = '<div class="cm-row">';
  html += `<div class="cm-cell header" title="Real \\ Predicho">Real↓ / Pred→</div>`;
  classes.forEach(c => {
    html += `<div class="cm-cell header" title="${c}">${c.substring(0, 6)}</div>`;
  });
  html += '</div>';

  // Filas
  classes.forEach(rowCls => {
    html += '<div class="cm-row">';
    html += `<div class="cm-cell row-header" title="${rowCls}">${rowCls.substring(0, 7)}</div>`;
    classes.forEach(colCls => {
      const val       = (matrix[rowCls] || {})[colCls] || 0;
      const isDiag    = rowCls === colCls;
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const bg        = isDiag
        ? `rgba(0, 229, 160, ${0.1 + intensity * 0.7})`   // diagonal: verde
        : val > 0
          ? `rgba(255, 95,  95, ${0.05 + intensity * 0.5})` // error: rojo
          : 'transparent';
      const color     = intensity > 0.5 ? '#fff' : 'var(--text-secondary)';
      html += `<div class="cm-cell" style="background:${bg};color:${color}" title="${rowCls}→${colCls}: ${val}">${val}</div>`;
    });
    html += '</div>';
  });

  container.innerHTML = html;
}

// ══════════════════════════════════════════════
// UTILIDADES
// ══════════════════════════════════════════════

function pct(val) {
  if (val === undefined || val === null) return '—';
  return (val * 100).toFixed(1) + '%';
}

function formatTimestamp(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('es-GT', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return iso;
  }
}