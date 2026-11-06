LearningProject: predicción de aceptación en telemarketing bancario

Repaso del flujo completo de un modelo de clasificación con el dataset UCI Bank Marketing (45.211 clientes, ~11,7% acepta el depósito a plazo).

Cómo reproducirlo
Crear el entorno e instalar: pip install -r requirements.txt
Colocar bank-full.csv en data/raw/ (no se sube al repo).
Generar los datos procesados: python src/preprocesamiento.py
Abrir los notebooks en orden.
Flujo
EDA (notebooks/01_exploracion.ipynb): tipos, nulos y análisis frente al objetivo.
Limpieza (src/preprocesamiento.py): reglas fijas, sin calcular nada con los datos (-1 a NaN en pdays, nuevas variables, eliminación de duration por fuga de información).
Modelo base (notebooks/03_modelo_base.ipynb): split estratificado 80/20; preprocesamiento (mediana, escalado, one-hot) ajustado solo con train dentro de un Pipeline; regresión logística evaluada con validación cruzada en train.
Resultados del modelo base (validación cruzada, train)
Modelo	Umbral	Recall	Precisión	AUC
Regresión logística	0,5	0,175	0,641	0,765
+ class_weight	0,5	0,626	0,270	0,765

class_weight y bajar el umbral llevan al mismo equilibrio entre recall y precisión: el AUC no cambió, así que el orden de los clientes es el mismo.

Se trabajó también con Random Forest y se obtuvieron los siguientes valores: ROC-AUC: 0.768 PR-AUC: 0.391

Se analizó modificando class_weight="balanced", pero se observa que lo empeora. Donde sí mejora es ajustando el hiperparámetro min_samples_leaf=10, con el que obtenemos: PR-AUC: 0.438, que supera a la regresión logística (0.402).

Aquí vemos que el modelo ofrece aproximadamente 3.7 veces más ventas por llamada que el azar (11.7%).

Se contemplaron diferentes umbrales para este modelo: umbral 0.5: recall=0.171 precisión=0.693 llamadas= 1044 ventas= 723 umbral 0.4: recall=0.242 precisión=0.605 llamadas= 1692 ventas= 1023 umbral 0.3: recall=0.393 precisión=0.508 llamadas= 3272 ventas= 1662 umbral 0.2: recall=0.531 precisión=0.425 llamadas= 5284 ventas= 2247 umbral 0.1: recall=0.726 precisión=0.249 llamadas= 12320 ventas= 3071

De lo anterior se indica que vale la pena hasta el umbral 0.2; luego es lo mismo que llamar al azar. Cabe mencionar que es importante evaluar el costo por llamada: si este no impacta demasiado, se favorece la elección del umbral 0.2, porque recoge más de la mitad de las ventas de train llamando a casi el 15% de la base. En conclusión, para conseguir mayor éxito se debe elegir Random Forest, pensando en concretar objetivos comerciales.

Se evaluaron los umbrales en test: umbral 0.5: recall=0.170 precisión=0.677 llamadas= 266 ventas= 180 umbral 0.4: recall=0.255 precisión=0.626 llamadas= 431 ventas= 270 umbral 0.3: recall=0.403 precisión=0.510 llamadas= 836 ventas= 426 umbral 0.2: recall=0.541 precisión=0.428 llamadas= 1337 ventas= 572 umbral 0.1: recall=0.735 precisión=0.256 llamadas= 3043 ventas= 778

Y se comparó la PR-AUC de test (0.443) con la de validación cruzada en train (0.438). En el umbral 0.2, train y test obtienen recall y precisión similares, lo que indica que no es un modelo sobreajustado.

## Conclusión de negocio

De lo anterior se indica que vale la pena hasta el umbral 0.2, luego es lo mismo que llamar al azar, cabe mencionar que es importante evaluar el costo por llamada, si esto no impacta demasiado se favorece la eleccion del umbral 0.2 porque recoge mas de la mitad de ventas de train llamando a casi el 15% de la base. En conclusion, en tema de conseguir mayor exito se debe elegir random forest pensando en concretar objetivos comerciales