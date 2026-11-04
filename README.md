# LearningProject: predicción de aceptación en telemarketing bancario

Repaso del flujo completo de un modelo de clasificación con el dataset UCI Bank Marketing (45.211 clientes, ~11,7% acepta el depósito a plazo).

## Cómo reproducirlo
1. Crear el entorno e instalar: `pip install -r requirements.txt`
2. Colocar `bank-full.csv` en `data/raw/` (no se sube al repo).
3. Generar los datos procesados: `python src/preprocesamiento.py`
4. Abrir los notebooks en orden.

## Flujo
1. **EDA** (`notebooks/01_exploracion.ipynb`): tipos, nulos y análisis frente al objetivo.
2. **Limpieza** (`src/preprocesamiento.py`): reglas fijas, sin calcular nada con los datos (`-1` a `NaN` en `pdays`, nuevas variables, eliminación de `duration` por fuga de información).
3. **Modelo base** (`notebooks/03_modelo_base.ipynb`): split estratificado 80/20; preprocesamiento (mediana, escalado, one-hot) ajustado solo con train dentro de un `Pipeline`; regresión logística evaluada con validación cruzada en train.

## Resultados del modelo base (validación cruzada, train)
| Modelo | Umbral | Recall | Precisión | AUC |
|---|---|---|---|---|
| Regresión logística | 0,5 | 0,175 | 0,641 | 0,765 |
| + `class_weight` | 0,5 | 0,626 | 0,270 | 0,765 |

`class_weight` y bajar el umbral llevan al mismo equilibrio entre recall y precisión: el AUC no cambió, así que el orden de los clientes es el mismo.

## Pendiente
- Random Forest y boosting.
- Elegir el umbral con el costo de una llamada y lo que deja un cliente que contrata.
- Evaluar en test una sola vez, con el modelo final.

