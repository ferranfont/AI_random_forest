# AI Trading Signal Detection - Random Forest Model

Sistema de detección de señales de iniciación de movimientos de precio utilizando Random Forest y análisis de volumen acelerado (Factor TPS).

---

## 📁 Estructura del Proyecto

```
AI_random_forest/
├── data/                           # Archivos CSV originales (time & sales)
├── data_ticks_per_second/          # CSV procesados con TPS calculado
├── utils/
│   └── clean_data_csv_to_ticks_per_second.py  # Procesamiento automático
├── train_initiation_model.py      # ⭐ SCRIPT 1: Entrenar modelo
├── forward_test_virgin_data.py     # ⭐ SCRIPT 2: Probar cualquier día
├── visualize_ai_signals.py         # 📦 Módulo interno (no ejecutar)
├── initiation_model.pkl            # Modelo entrenado (Random Forest)
└── outputs/                        # Gráficos HTML y CSVs de señales
```

---

## 🎯 ¿Qué Script Usar y Cuándo?

### ⭐ **SOLO NECESITAS 2 SCRIPTS:**

| Script | Cuándo Usarlo | Qué Hace |
|--------|---------------|----------|
| **`train_initiation_model.py`** | **Una sola vez** (o cuando quieras re-entrenar) | Crea el archivo `initiation_model.pkl` con el cerebro del modelo |
| **`forward_test_virgin_data.py`** | **Siempre** que quieras probar un día nuevo | Procesa datos + genera gráfico HTML + CSV de señales |

### 📦 **NO EJECUTAR DIRECTAMENTE:**

- **`visualize_ai_signals.py`**: Es un módulo/librería que usa `forward_test_virgin_data.py` internamente
- **`utils/clean_data_csv_to_ticks_per_second.py`**: Se ejecuta automáticamente desde `forward_test_virgin_data.py`

---

## 🚀 Flujo de Trabajo Completo

### 1️⃣ **ENTRENAR EL MODELO** (Solo una vez)

```bash
python train_initiation_model.py
```

**¿Qué hace?**
- Lee datos históricos de `data_ticks_per_second/tps_time_and_sales_nq_20251103.csv`
- Crea features (lags, medias, velocidad de precio)
- Etiqueta señales de "iniciación" usando heurísticas
- Entrena Random Forest (100 árboles)
- **Guarda:** `initiation_model.pkl` ← El cerebro del modelo

**Output:**
```
✅ initiation_model.pkl creado
📊 Métricas de evaluación mostradas en consola
```

---

### 2️⃣ **PROBAR CUALQUIER DÍA** (Uso diario)

**Edita la línea 6 de `forward_test_virgin_data.py`:**

```python
CSV_FILE = "time_and_sales_nq_20251104"  # ← Cambia la fecha aquí
```

**Ejecuta:**

```bash
python forward_test_virgin_data.py
```

**¿Qué hace automáticamente?**
1. ✅ Busca el archivo procesado en `data_ticks_per_second/`
2. ❌ Si NO existe → Lo procesa desde `data/` usando el algoritmo correcto
3. 🧠 Carga el modelo `initiation_model.pkl`
4. 🎨 Genera señales y crea gráfico interactivo
5. 💾 Guarda en `outputs/`:
   - `ai_signals_YYYYMMDD.html` (gráfico interactivo)
   - `ai_signals_YYYYMMDD.csv` (señales detectadas)

**Output:**
```
✅ Found processed TPS file: data_ticks_per_second/tps_time_and_sales_nq_20251104.csv
🎨 Running AI Model Visualization...
📊 Chart saved to: outputs/ai_signals_20251104.html
📄 Signals saved to: outputs/ai_signals_20251104.csv
```

---

## 📊 Entendiendo el Modelo

### ¿Qué es Factor TPS?

**Factor TPS = `volumen_ventana × ticks_por_segundo`**

- Detecta **aceleraciones de volumen** (no solo volumen alto)
- Valores altos (>4000) indican potencial inicio de movimiento fuerte
- Se calcula automáticamente en el procesamiento

### ¿Cómo se Etiquetan las Señales?

El modelo aprende de señales definidas por **heurísticas** (no hay labels reales):

```python
# En train_initiation_model.py, función define_labels()
tps_threshold = 4000         # Factor TPS alto
price_move_threshold = 3.5   # Movimiento mínimo de precio (ticks)
future_window = 10           # Segundos hacia adelante
```

**Una señal de "iniciación" es:**
- ✅ Factor TPS > 4000 (aceleración)
- ✅ Precio se mueve ≥3.5 ticks en los próximos 10 segundos

---

## ⚙️ Configuración Avanzada

### Cambiar Sensibilidad del Modelo

**Edita `train_initiation_model.py`, línea 87:**

```python
def define_labels(df, 
    tps_threshold=4000,        # ↑ Más selectivo | ↓ Más señales
    price_move_threshold=3.5,  # ↑ Solo movimientos grandes | ↓ Más señales
    future_window=10):         # Ventana de tiempo (segundos)
```

**Después de cambiar, RE-ENTRENAR:**
```bash
python train_initiation_model.py
```

### Cambiar Archivo de Entrenamiento

**Edita `train_initiation_model.py`, línea 166:**

```python
CSV_PATH = r"d:\PYTHON\ALGOS\AI_random_forest\data_ticks_per_second\tps_time_and_sales_nq_20251103.csv"
```

---

## 📈 Formato de Datos

### CSV Raw (Input en `data/`)
```csv
Timestamp;Precio;Volumen;Lado;Bid;Ask
2025-11-03 06:00:05.920;26085,0;1;ASK;26084,75;26085,25
```

### CSV Procesado (Output en `data_ticks_per_second/`)
```csv
Timestamp;Precio;Volumen;Lado;Bid;Ask;window_vol;tps_window;factor_tps
2025-11-03 06:00:05.920;26085,0;1;ASK;26084,75;26085,25;150;13,5;2025,0
```

**Nota:** Separador `;` y decimal `,` (formato europeo)

---

## 🎨 Visualización de Señales

El gráfico HTML generado muestra:

- **Línea gris:** Precio del activo
- **Puntos verdes:** Señales de compra (movimiento alcista detectado)
- **Puntos rojos:** Señales de venta (movimiento bajista detectado)

**Características:**
- ✅ Interactivo (zoom, pan)
- ✅ Sin hover info (limpio)
- ✅ Sin grid vertical
- ✅ Abre directamente en navegador

---

## 🛠️ Instalación

### Dependencias

```bash
pip install pandas numpy scikit-learn joblib matplotlib plotly
```

### Estructura de Carpetas Requerida

```
AI_random_forest/
├── data/                    # Coloca aquí tus CSVs raw
├── data_ticks_per_second/   # Se crea automáticamente
└── outputs/                 # Se crea automáticamente
```

---

## 🔍 Troubleshooting

### ❌ "No initiation signals found"

**Solución:**
- Reducir `tps_threshold` (ej: 3000 en vez de 4000)
- Reducir `price_move_threshold` (ej: 2.5 en vez de 3.5)
- Verificar que el CSV tenga suficientes datos (>100,000 filas)

### ❌ "Error loading data"

**Solución:**
- Verificar formato del CSV: separador `;` y decimal `,`
- Comprobar nombres de columnas (español con espacios)
- Verificar que el archivo existe en `data/`

### ❌ "Model not found"

**Solución:**
```bash
python train_initiation_model.py  # Crear el modelo primero
```

### ❌ "Raw CSV file not found"

**Solución:**
- Verificar que el archivo existe en `data/time_and_sales_nq_YYYYMMDD.csv`
- Verificar que `CSV_FILE` en `forward_test_virgin_data.py` tiene el nombre correcto (sin extensión .csv)

---

## 📊 Resultados Típicos

**Última ejecución (20251103):**
- **Datos procesados:** 405,719 samples (1 por segundo)
- **Señales detectadas:** 672 (~0.17% de los datos)
- **Factor TPS rango:** 0 - 6,250
- **Modelo:** Random Forest (100 estimators, class_weight='balanced')
- **Precisión:** ~75% en test set

---

## 💡 Conceptos Clave

### ¿Por qué Random Forest?

- ✅ Maneja bien features no lineales (lags, medias)
- ✅ Robusto a outliers
- ✅ No requiere normalización
- ✅ Proporciona importancia de features

### ¿Por qué Etiquetado Heurístico?

- No hay "ground truth" real de señales de iniciación
- Las heurísticas capturan conocimiento de trading
- El modelo aprende **patrones** que preceden a estas condiciones

### ¿Qué es "Virgin Data"?

Datos **completamente nuevos** que el modelo nunca vio durante entrenamiento:
- Fechas diferentes
- Condiciones de mercado diferentes
- Prueba real de generalización del modelo

---

## 📧 Soporte

Para dudas sobre la implementación, revisar los comentarios en cada script.

**Archivos clave:**
- `train_initiation_model.py` - Lógica de entrenamiento
- `forward_test_virgin_data.py` - Lógica de testing
- `utils/clean_data_csv_to_ticks_per_second.py` - Cálculo de Factor TPS
