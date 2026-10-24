import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/bank-full.csv")
PROCESSED = Path("data/processed/bank-full_cleaned.csv") 



def cargar(ruta):
    df= pd.read_csv(ruta, sep=";")
    return df


def limpiar(df):
    df = df.copy()
    df['aceptacion'] = df['y'].replace(['no','yes'],['0','1'])
    df['aceptacion']=df['aceptacion'].astype(int)
    df['contactado_antes'] = [1 if x != -1 else 0 for x in df["pdays"]]
    df["pdays"] = df["pdays"].replace(-1,np.nan)
    df["n_creditos"] = (df["housing"] == "yes").astype(int) + (df["loan"] == "yes").astype(int)
    df["en_sobregiro"] = [1 if x < 0 else 0 for x in df["balance"]]
    df["rango"] = pd.cut(df["age"], bins=[ 0,30, 50, 100], labels=["<=30", "31-50", ">50"])
    df.drop(columns=["duration","y","housing","loan", "age"],inplace=True)
    return df



def guardar(df, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)  # crea data/processed si no existe
    df.to_csv(ruta, index=False)


if __name__ == "__main__":
    df = cargar(RAW)
    df = limpiar(df)
    guardar(df, PROCESSED)
    print(df.shape)
    print(df['contactado_antes'].value_counts())
    print(df['n_creditos'].value_counts())
    print(df['en_sobregiro'].value_counts())
    print(df.columns)
    print(df["pdays"].isna().sum())
    print(df["aceptacion"].mean())
    print(df.shape)


    


