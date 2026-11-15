# LearningProject: predicción de aceptación en telemarketing bancario

Repaso del flujo completo de un modelo de clasificación con el dataset UCI Bank Marketing (45.211 clientes, ~11,7% acepta el depósito a plazo).

## Problema de negocio
El banco tiene un problema de conversión: sobre su base total, solo el 11,7% acepta. Por eso surge la necesidad de elegir mejor a quién llamar. El objetivo de este desarrollo es apoyarse en un modelo de machine learning para mejorar la tasa de aceptación.

## Cómo reproducirlo
Todos los comandos se corren desde la raíz del proyecto (la carpeta `LearningProject`).

1. Clonar el repositorio y entrar a la carpeta:
   ```
   git clone https://github.com/Darkesthj0399/LearningProject.git
   cd LearningProject
   ```
2. Crear y activar el entorno virtual:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac o Linux
   ```
3. Instalar las librerías: `pip install -r requirements.txt`
4. Descargar el dataset desde [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) y colocar `bank-full.csv` en `data/raw/` (los datos no se suben al repo).
5. Generar los datos procesados: `python src/preprocesamiento.py`
6. Entrenar y guardar el modelo: `python src/train.py`. Muestra la PR-AUC en test (0,467) y guarda el modelo en `models/modelo.joblib`.
7. Predecir sobre clientes nuevos: `python src/predict.py` (ver la sección siguiente).
8. Opcional: abrir los notebooks en orden para revisar el análisis completo.

## Uso del modelo con clientes nuevos
- **Entrada:** `data/raw/clientes_nuevos.csv`, con el mismo formato que `bank-full.csv` (separador `;`), pero sin las columnas `y` ni `duration`, que no se conocen antes de llamar.
- **Proceso:** `predict.py` aplica la misma limpieza que el entrenamiento (`limpiar()` de `preprocesamiento.py`), carga el modelo y calcula la probabilidad de compra de cada cliente.
- **Salida:** `data/processed/predicciones.csv`, ordenado de mayor a menor probabilidad, con dos columnas nuevas: `probabilidad` y `llamar` (1 = llamar).
- **Umbral:** 0,2, definido en la constante `UMBRAL` al inicio de `predict.py`. Cambiarlo no requiere reentrenar.

## Estructura
```
data/raw/            datos crudos (no se suben)
data/processed/      datos limpios y predicciones (no se suben)
models/              modelo entrenado (no se sube; se genera con train.py)
notebooks/           análisis: EDA, limpieza, modelos
src/
  preprocesamiento.py   reglas fijas de limpieza
  train.py              entrena el modelo elegido y lo guarda
  predict.py            aplica el modelo y el umbral a clientes nuevos
```

## Flujo
1. **EDA** (`notebooks/01_exploracion.ipynb`): tipos, nulos y análisis frente al objetivo.
2. **Limpieza** (`src/preprocesamiento.py`): reglas fijas, sin calcular nada con los datos (`-1` a `NaN` en `pdays`, nuevas variables, eliminación de `duration` por fuga de información).
3. **Modelo base** (`notebooks/03_modelo_base.ipynb`): split estratificado 80/20; preprocesamiento (mediana, escalado, one-hot) ajustado solo con train dentro de un `Pipeline`; regresión logística evaluada con validación cruzada en train.
4. **Comparación de modelos:** regresión logística, Random Forest y boosting (`HistGradientBoostingClassifier`), comparados con PR-AUC en validación cruzada. Se usa PR-AUC porque con ~11,7% de positivos mide mejor qué tan arriba de la lista quedan los clientes que compran.
5. **Ajuste de hiperparámetros:** `min_samples_leaf` a mano en Random Forest y con `GridSearchCV` en boosting.
6. **Umbral:** en vez de un solo número, una tabla de campaña (llamadas y ventas por umbral) elegida con validación cruzada.
7. **Evaluación final en test** e **importancia de variables** por permutación.
8. **Empaquetado:** el modelo elegido pasa a `train.py` y `predict.py`, reproducibles desde cero con este README.

## Resultados

### Comparación de modelos (PR-AUC, validación cruzada en train)
| Modelo | PR-AUC |
|---|---|
| Regresión logística | 0,402 |
| Random Forest (por defecto) | 0,391 |
| Random Forest (`min_samples_leaf=10`) | 0,438 |
| Boosting (por defecto) | 0,442 |
| **Boosting ajustado (`min_samples_leaf=50`)** | **0,450** |

Azar: 0,117 (porcentaje de clientes que compran). `class_weight` no mejoró el ranking en ningún modelo; en Random Forest lo empeoró.

### Modelo elegido: boosting ajustado
- PR-AUC en validación cruzada: **0,450**. En test: **0,467**. Son consistentes, así que el modelo no está sobreajustado.
- La diferencia con Random Forest es pequeña (0,012), pero boosting ganó tanto por defecto como ajustado.
- **Nota sobre test:** se usó dos veces (primero con Random Forest, luego con boosting). La elección del modelo y del umbral se hizo siempre con validación cruzada; test solo confirmó.

### Tabla de campaña (test, 9.043 clientes)
| Umbral | Llamadas | Ventas | Precisión | Recall |
|---|---|---|---|---|
| 0,5 | 411 | 258 | 0,628 | 0,244 |
| 0,4 | 645 | 362 | 0,561 | 0,342 |
| 0,3 | 914 | 483 | 0,528 | 0,457 |
| 0,2 | 1.216 | 558 | 0,459 | 0,527 |
| 0,1 | 2.585 | 742 | 0,287 | 0,701 |

Por debajo de 0,2 las llamadas extra convierten casi como el azar.

### Importancia de variables (caída de PR-AUC en test al desordenar cada variable)
| Variable | Importancia |
|---|---|
| poutcome (resultado de la campaña anterior) | 0,134 |
| month | 0,121 |
| contact | 0,076 |
| day | 0,065 |
| pdays | 0,025 |
| n_creditos | 0,024 |
| Resto (campaign, previous, marital, balance, education, job, rango de edad, en_sobregiro, contactado_antes, default) | < 0,01 |

Las variables importantes para el modelo son poutcome, month, contact y day; las demás tienen un impacto casi despreciable. Month y day pueden estar aprendiendo el calendario de las campañas de estos datos, lo que no implica que en esos meses o días se concreten más ventas. Pesa más la relación previa con el cliente y cómo y cuándo se lo contacta que su perfil personal: la edad, el trabajo o el saldo casi no influyen.

## Conclusión de negocio
El mejor modelo es el de boosting con umbral 0,2: llamando al 13,4% de la base se capta el 52,7% de los compradores (datos de test), y el 46% de las llamadas termina en venta, casi 4 veces más que llamar al azar (11,7%). Se recomiendan umbrales de 0,2 o más, porque las llamadas extra que se agregan al bajar de 0,2 convierten casi igual que el azar.

## Limitaciones
- El dataset no tiene un identificador de cliente: la lista de predicciones indica qué filas llamar, no a quién. En producción, la entrada debe traer un `id_cliente` que pase al resultado sin entrar al modelo.
- El modelo se evaluó con datos de un solo período; month y day podrían no generalizar a campañas en otras fechas.

## Próximos pasos
- Simplificar el modelo quitando las variables que no aportan.
- Agregar el `id_cliente` a la entrada y la salida de `predict.py`.
- Registro de experimentos y despliegue (MLflow, API, Docker, nube): fuera del alcance de este proyecto; se trabajarán en un pipeline de datos en AWS.