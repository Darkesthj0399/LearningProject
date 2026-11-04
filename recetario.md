## .iloc vs .loc
- Etiqueta (índice) = posición vieja. Tras train_test_split queda desordenada (27164, 25950…).
- Posición = orden actual (0, 1, 2…). No se imprime.
- `.iloc[i]` → posición nueva | `.loc[i]` → etiqueta
- **Ojo con:** proba_cv (numpy) solo tiene posiciones y sigue el orden de y_train → cruzar con `.iloc`. Con `.loc` o `[]` se mezclan clientes sin dar error.

## precision@k
**Para qué:** precisión cuando el negocio fija k llamadas.
```python
idx = np.argsort(proba)[::-1][:k]   # posiciones de los k con mayor probabilidad
y_train.iloc[idx].mean()            # proporción de "sí" en el top k
```
- **Ojo con:** la media de 0/1 = proporción de "sí".

## ROC-AUC y PR-AUC
**Para qué:** comparar modelos sin depender del umbral. Reciben probabilidades, no etiquetas.
```python
roc_auc_score(y, proba)            # azar = 0.5
average_precision_score(y, proba)  # azar = % de positivos (0.117)
```
- **Qué mide:** ordenas a los clientes de mayor a menor probabilidad de compra; mientras más arriba queden los que sí compran, más alta la PR-AUC.
- **Para el negocio:** mayor PR-AUC = más ventas por llamada que los otros modelos (0.438 vs 0.117 al azar ≈ 3.7 veces más).
- **Ojo con:** con clases desbalanceadas, PR-AUC es más informativa. Es un promedio de toda la lista: con un presupuesto fijo de llamadas, decide precision@k.

## Tamaño de un arreglo
`len(arr)` o `arr.shape[0]` (`.length` no existe)

## min_samples_leaf (Random Forest)
- Mínimo de clientes en cada grupo final (hoja). La probabilidad = % de compradores de ese grupo.
- Muy bajo (1) → memoriza (sobreajuste). Muy alto → grupos demasiado mezclados (subajuste).
- Probar pocos valores separados (10, 20, 50) y quedarse con el mejor en CV. Aquí ganó 10 (PR-AUC 0.438).

## Funciones
```python
def precision_at_k(proba, k):
    ...
    return clientes, prec            # lo que va después del return no se ejecuta

clientes, prec = precision_at_k(proba_cv, 500)   # recibe los valores en el mismo orden
print(f"RL: {clientes} clientes, precisión {prec:.3f}")   # f-string; :.3f = 3 decimales
```
- **Ojo con:** todo lo que la función necesita entra por parámetros, no de variables de afuera.

---
# De clases anteriores

## train_test_split estratificado
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```
- `stratify=y` mantiene el % de "sí" en train y test. Va antes de cualquier paso que aprenda de los datos.

## Pipeline + ColumnTransformer
- `ColumnTransformer`: pasos en paralelo, uno por grupo de columnas (num, cat…).
- `Pipeline`: pasos en secuencia (preprocesar → modelo).
- **Ojo con:** `fit` aprende solo de train; `transform` aplica eso a test. Dentro de `cross_val_predict` se hace solo en cada fold.

## predict_proba y umbral
```python
proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)
```
- `[:, 1]` = columna de probabilidad de "sí".
- Bajar el umbral sube recall y baja precisión.

## class_weight="balanced"
- Da más peso a los errores en la clase pequeña. En la práctica empuja las probabilidades hacia arriba ≈ bajar el umbral.
- **Ojo con:** no mejora el ranking (AUC igual en RL; en RF lo empeoró).


## Umbral → tabla de campaña
**Para qué:** traducir el modelo en opciones de negocio (cuántos llamar, cuántas ventas).
```python
for umbral in [0.5, 0.4, 0.3, 0.2, 0.1]:
    p = (proba >= umbral).astype(int)   # 1 = lo llamo, 0 = no lo llamo
    print(f"umbral {umbral}: llamadas={p.sum()} ventas={(p * y).sum()}")
```
- `p.sum()` = llamadas. `(p * y).sum()` = ventas (1 solo si lo llamé Y compró).
- **Ojo con:** si las llamadas extra de un salto rinden como el % de positivos (azar), ese umbral ya no aporta.
- Al negocio se le da en proporciones: "llamando al 15% de la base capturamos el 54% de los compradores".

## CV vs test
- **CV (dentro de train):** para elegir modelo e hiperparámetros. Cada fold se evalúa con clientes que ese modelo no vio.
- **Test:** prueba final, una sola vez. Después no se ajusta nada.
- **Sobreajuste:** modelo final prediciendo sobre su propio train (nota inflada) vs CV (nota honesta). Si la inflada está muy arriba, memorizó.
```python
pipe.fit(X_train, y_train)                    # modelo final, con todo train
proba_test = pipe.predict_proba(X_test)[:, 1] # el pipeline solo aplica (transform) en test
```