# 1. Imports
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score                                  # arriba, con los demás imports


# 2. Rutas y constantes: datos procesados, dónde guardar el modelo, columnas numéricas y categóricas
PROCESSED = Path("data/processed/bank-full_cleaned.csv") 
cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome', 'month', 'n_creditos', 'rango', 'default']
num_cols = ['balance', 'campaign', 'previous', 'day']
MODELO = Path("models/modelo.joblib")          

# 3. Leer el CSV procesado y separar X e y

df = pd.read_csv(PROCESSED)
X = df.drop(columns='aceptacion')
y = df['aceptacion']

# 4. Split estratificado 80/20 con random_state=42

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)

# 5. Armar el Pipeline (ColumnTransformer + boosting con min_samples_leaf=50)


pre_dias = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler()),
])

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("dias", pre_dias, ["pdays"]),
    ("cat", OneHotEncoder(handle_unknown="ignore",drop="if_binary"), cat_cols),
    ("bin", "passthrough", ['contactado_antes','en_sobregiro']),
],sparse_threshold=0)

# 6. Entrenar con todo train

pipe_hgb = Pipeline([
    ("pre", preprocesador),
    ("modelo", HistGradientBoostingClassifier(random_state=42, learning_rate=0.1,max_iter=100, min_samples_leaf=50)),
])

pipe_hgb.fit(X_train, y_train)
# 7. PR-AUC en test
proba_hgb_test = pipe_hgb.predict_proba(X_test)[:, 1]
print("PR-AUC HGB:", round(average_precision_score(y_test, proba_hgb_test), 3))
# 8. Guardar el Pipeline con joblib

MODELO.parent.mkdir(parents=True, exist_ok=True)   # crea la carpeta models/ si no existe
joblib.dump(pipe_hgb, MODELO)                      # guarda el Pipeline entrenado
print("Modelo guardado en", MODELO)