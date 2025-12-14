# AI Trading Signal Detection - Random Forest Model

Sistema de detección de señales de iniciación de movimientos de precio utilizando Random Forest y análisis de volumen acelerado (Factor TPS).

---

## 📁 Estructura del Proyecto

```
AI_random_forest/
├── data/                           # Archivos CSV originales (time & sales)
├── data_ticks_per_second/          # CSV procesados con TPS calculado
├── utils/
│   └── clean_data_csv_to_ticks_per_second.py  # Procesamiento de datos
├── train_initiation_model.py      # Entrenamiento del modelo
├── visualize_ai_signals.py         # Visualización de señales detectadas
├── forward_test_virgin_data.py     # Test con datos nuevos
├── initiation_model.pkl            # Modelo entrenado (Random Forest)
└── outputs/                        # Gráficos y CSVs de resultados
```

---

## 🔄 Pipeline de Datos

### 1. Preprocesamiento de Datos

**Script:** `utils/clean_data_csv_to_ticks_per_second.py`

Convierte archivos raw CSV (time & sales) a datos agregados por segundo con Factor TPS calculado:

```bash
python utils/clean_data_csv_to_ticks_per_second.py
```

**Input:** `data/time_and_sales_nq_*.csv`  
**Output:** `data_ticks_per_second/tps_time_and_sales_nq_*.csv`

**Columnas generadas:**
- `Timestamp` - Marca temporal
- `Precio` - Último precio del segundo
- `Volumen` - Volumen agregado
- `Lado` - BID/ASK
- `Bid/Ask` - Spread
- `window_vol` - Volumen en ventana
- `tps_window` - TPS en ventana
- `factor_tps` - **Factor TPS = window_vol × tps_window** (métrica clave)

---

### 2. Entrenamiento del Modelo

**Script:** `train_initiation_model.py`

Entrena un modelo Random Forest para detectar señales de "iniciación" basadas en:
- Alto Factor TPS (aceleración de volumen)
- Movimiento de precio significativo posterior

```bash
python train_initiation_model.py
```

**Features generadas:**
- Lags de Factor TPS (1-5 periodos)
- Media y desviación estándar (ventana de 5)
- Velocidad de precio

**Etiquetado heurístico:**
- `tps_threshold = 4000` (umbral alto de TPS)
- `price_move_threshold = 3.5` (ticks de movimiento)
- `future_window = 10` (segundos hacia adelante)

**Output:** `initiation_model.pkl` (modelo entrenado)

---

### 3. Visualización de Señales

**Script:** `visualize_ai_signals.py`

Genera gráficos interactivos HTML con Plotly mostrando:
- Línea de precio (gris)
- Señales de compra (verde) - movimiento alcista
- Señales de venta (rojo) - movimiento bajista

```bash
python visualize_ai_signals.py
```

**Requiere:** `initiation_model.pkl` (modelo pre-entrenado)  
**Output:** `outputs/ai_signals_chart.html` + CSV con señales

**Características del gráfico:**
- Interactivo (zoom, pan)
- Sin hover info (limpio)
- Sin grid vertical
- Colores diferenciados por dirección

---

### 4. Forward Test (Datos Nuevos)

**Script:** `forward_test_virgin_data.py`

Prueba el modelo con datos completamente nuevos (no vistos durante entrenamiento):

```bash
python forward_test_virgin_data.py
```

**Input:** CSV raw desde `data/`  
**Process:** Calcula TPS on-the-fly, aplica modelo  
**Output:** `outputs/ai_signals_chart_virgin.html`

---

## 🚀 Uso Rápido

### Primera vez (Setup completo)

```bash
# 1. Procesar datos raw
python utils/clean_data_csv_to_ticks_per_second.py

# 2. Entrenar modelo
python train_initiation_model.py

# 3. Visualizar señales
python visualize_ai_signals.py
```

### Uso regular (modelo ya entrenado)

```bash
# Solo visualizar señales con modelo existente
python visualize_ai_signals.py

# O test con datos nuevos
python forward_test_virgin_data.py
```

---

## ⚙️ Configuración

### Ajustar Sensibilidad del Modelo

Edita `train_initiation_model.py`, función `define_labels()`:

```python
def define_labels(df, 
    tps_threshold=4000,        # ↑ Más selectivo | ↓ Más señales
    price_move_threshold=3.5,  # ↑ Movimientos grandes | ↓ Más señales
    future_window=10):         # Segundos hacia adelante
```

**Después de cambiar, re-entrenar:**
```bash
python train_initiation_model.py
```

---

## 📊 Formato de Datos

### CSV de Entrada (Raw)
```csv
Timestamp;Precio;Volumen;Lado;Bid;Ask
2025-11-03 06:00:05.920;26085,0;1;ASK;26084,75;26085,25
```

### CSV Procesado (TPS)
```csv
Timestamp; Precio; Volumen; factor_tps
2025-11-03 06:00:05.920; 26085,0; 1; 2,08
```

**Nota:** Separador `;` y decimal `,` (formato europeo)

---

## 📈 Resultados del Modelo

**Última ejecución:**
- **Datos procesados:** 405,719 samples
- **Señales detectadas:** 672 (~0.17%)
- **Factor TPS rango:** 0 - 6,250
- **Modelo:** Random Forest (100 estimators, balanced)

---

## 🛠️ Dependencias

```python
pandas
numpy
scikit-learn
joblib
matplotlib
plotly
```

Instalar:
```bash
pip install pandas numpy scikit-learn joblib matplotlib plotly
```

---

## 📝 Notas Importantes

1. **Factor TPS:** Métrica propietaria = `volumen × ticks_por_segundo`
   - Detecta aceleraciones de volumen
   - Valores altos (>4000) indican potenciales iniciaciones

2. **Etiquetado Heurístico:** El modelo aprende de señales definidas manualmente
   - No es supervisado puro (no hay labels reales)
   - Ajusta umbrales según mercado/instrumento

3. **Archivos Grandes:** Los CSV pueden ser muy pesados
   - `data_ticks_per_second/` contiene datos agregados (más ligeros)
   - `data/` contiene raw tick-by-tick (pesados)

---

## 🔍 Troubleshooting

### "No initiation signals found"
- Reducir `tps_threshold` o `price_move_threshold`
- Verificar que el CSV tenga suficientes datos

### "Error loading data"
- Verificar formato del CSV (`;` separador, `,` decimal)
- Comprobar nombres de columnas (español con espacios)

### "Model not found"
- Ejecutar primero `python train_initiation_model.py`
- Verificar que `initiation_model.pkl` exista

---

## 📧 Contacto

Para ajustes del modelo o dudas sobre la implementación, revisar los comentarios en cada script.
