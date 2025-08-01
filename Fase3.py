## 1. Importar librerías y cargar datos necesarios
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1.1 – Ruta del modelo entrenado de Fase 2
MODEL_PATH = "rnn_demand_model.h5"
DATA_PATH = "DataFinal.csv"

app = FastAPI()

# 1.2 – Cargar DataFrame original de secuencias para tener acceso a 
#       todas las filas con lag features (DataFinal.csv). 
#       Necesitamos este DF para extraer lags históricos hasta 14 días antes de “date”.
DF_SEQ = pd.read_csv(DATA_PATH, parse_dates=["fecha_final_ventana"])



# 1.3 – Cargar lista de columnas finales (en caso de que las tengas en texto)
# Si guardaste columnas_finales.txt, podrías leerlo aquí; 
# de lo contrario, usamos directamente DF_SEQ.columns
lag_cols = [c for c in DF_SEQ.columns if "_t-" in c]  # igual que en Fase 2
lag_cols = sorted(lag_cols)  # opcional: asegúrate de orden por t-1, t-2, …

print("Total de filas en DF_SEQ:", len(DF_SEQ))
print("Número de columnas de lag:", len(lag_cols))



def get_14lags_for(product_id: str, date_str: str):
    """
    Dado un product_id y una fecha en 'YYYY-MM-DD', 
    retorna un array de forma (1, 14, n_feats) con dtype=float32, 
    o None si no hay suficiente historial.
    """
    # 2.1 – Convertir date_str a datetime
    fecha_obj = pd.to_datetime(date_str)

    # 2.2 – Filtrar DF_SEQ por este product_id
    df_prod = DF_SEQ[DF_SEQ["product_id"].astype(str) == str(product_id)].copy()
    if df_prod.empty:
        return None  # producto no existe

    # 2.3 – Quedarnos solo con filas cuya fecha_final_ventana < fecha_obj
    df_prod_hist = df_prod[df_prod["fecha_final_ventana"] < fecha_obj]
    if df_prod_hist.empty:
        return None

    # 2.4 – Intentar encontrar la fila justo en fecha_obj - 1 día
    fecha_obj_menos1 = fecha_obj - pd.Timedelta(days=1)
    if fecha_obj_menos1 in set(df_prod_hist["fecha_final_ventana"]):
        fila = df_prod_hist[df_prod_hist["fecha_final_ventana"] == fecha_obj_menos1].iloc[-1]
    else:
        # Si no existe exactamente fecha_obj-1, tomamos la última fila anterior
        fila = df_prod_hist.iloc[-1]

    # 2.5 – Extraer valores de lag_cols como 1D
    arr_1d = fila[lag_cols].values  # tipo original puede ser object

    # 2.6 – Convertir a float32
    arr_1d = arr_1d.astype(np.float32)

    # 2.7 – Reshape a (1, 14, n_feats)
    n_total_feats = len(lag_cols)
    n_timesteps = 14
    n_feats = n_total_feats // n_timesteps
    arr_reshaped = arr_1d.reshape((1, n_timesteps, n_feats))

    return arr_reshaped





# 3.1 – Cargar el modelo una única vez en memoria, sin compilar (compile=False)
_model = load_model(MODEL_PATH, compile=False)

# 3.1.1 – Recompilar el modelo con loss y métricas como strings
_model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mae"]
)

def predict_demand(product_id: str, date: str, verbose: bool = False):
    """
    Parámetros:
      - product_id: ID del producto (cadena o número según cómo esté en DF_SEQ)
      - date: Fecha objetivo en formato 'YYYY-MM-DD'
    Retorno:
      - float: predicción de ventas (sale_amount) para el día “date” 
      - None si no hay suficiente historial o producto no existe
    """
    # 3.2 – Obtener array de lags
    seq_14 = get_14lags_for(product_id, date)
    if seq_14 is None:
        if verbose:
            print(f"No hay suficiente historial para product_id={product_id} antes de {date}.")
        return None

    # 3.3 – Normalización / escalamiento 
    # (Si en Fase 2 aplicaste algún escalador, aquí cargarías y lo aplicarías. 
    #  En este ejemplo asumimos que DF_SEQ ya está normalizado.)

    # 3.4 – Hacer la predicción
    pred = _model.predict(seq_14, verbose=0)

    # 3.5 – Retornar un float redondeado a 2 decimales
    return float(np.round(pred[0, 0], 2))








# Esquema Pydantic para la request
class PredictRequest(BaseModel):
    product_id: str
    date: str  # 'YYYY-MM-DD'

class PredictResponse(BaseModel):
    product_id: str
    date: str
    prediction: float

@app.post("/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    pred = predict_demand(req.product_id, req.date)
    if pred is None:
        raise HTTPException(status_code=404, detail="No hay suficiente historial o producto no existe")
    return PredictResponse(product_id=req.product_id, date=req.date, prediction=pred)

# Para ejecutar: uvicorn Fase3:app --reload --port 8000
