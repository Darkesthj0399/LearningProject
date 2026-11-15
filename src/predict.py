# 1. Imports (joblib, pandas, Path y tu función limpiar)
import joblib
import pandas as pd
from pathlib import Path
from preprocesamiento import cargar, limpiar

# 2. Constantes: ruta del modelo, ruta de clientes nuevos, ruta de salida y UMBRAL
MODELO = Path("models/modelo.joblib")       
UMBRAL =0.2   
NUEVOS = Path("data/raw/clientes_nuevos.csv")        # entrada: clientes de la campaña
SALIDA = Path("data/processed/predicciones.csv")
# 3. Cargar los clientes nuevos y pasarlos por limpiar()
df = cargar(NUEVOS)
df = limpiar(df)
# 4. Cargar el modelo con joblib.load
modelo = joblib.load(MODELO)

# 5. Calcular la probabilidad de cada cliente
df["probabilidad"] = modelo.predict_proba(df)[:, 1]

# 6. Aplicar el umbral: columna "llamar" (1 = llamar)
df["llamar"] = (df["probabilidad"] >= UMBRAL).astype(int)

# 7. Ordenar de mayor a menor probabilidad y guardar el CSV
df = df.sort_values("probabilidad", ascending=False)   # el más probable primero
SALIDA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SALIDA, index=False)
# 8. Imprimir cuántos clientes hay que llamar
print(f"Clientes a llamar: {df['llamar'].sum()} de {len(df)} (umbral {UMBRAL})")