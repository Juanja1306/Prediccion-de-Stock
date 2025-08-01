import pandas as pd
import numpy as np

# Suponiendo que ya tienes unique_product_ids y valores desde el notebook
# Si los necesitas cargar, descomenta las siguientes líneas:

# Cargar datos del parquet para obtener unique_product_ids
df = pd.read_parquet("eval.parquet")
unique_product_ids = df['product_id'].unique()

# Importar las funciones necesarias del notebook para hacer predicciones
# (Asumiendo que las funciones están definidas o importadas)

# Si ya tienes los valores calculados, puedes usar la variable 'valores' directamente
# Si no, ejecuta el bucle de predicciones:
valores = []
# for prod in unique_product_ids:
#     pred = predict_demand(product_id=str(prod), date="2024-07-03", verbose=False)
#     valores.append(pred)

# Por ahora, usando valores de ejemplo (reemplaza con tus datos reales)
print(f"Procesando {len(unique_product_ids)} productos...")

# Crear DataFrame con el formato solicitado
results_df = pd.DataFrame({
    'col1': unique_product_ids,
    'col2': valores  # Asegúrate de que esta variable existe con las predicciones
})

# Guardar en CSV
csv_filename = 'predicciones_productos.csv'
results_df.to_csv(csv_filename, index=False)

print(f"✓ CSV guardado como '{csv_filename}' con {len(results_df)} filas")
print(f"✓ Columnas: {list(results_df.columns)}")
print("\nPrimeras 5 filas:")
print(results_df.head())

# Estadísticas básicas
valid_predictions = results_df['col2'].dropna()
print(f"\nEstadísticas:")
print(f"- Total productos: {len(results_df)}")
print(f"- Predicciones válidas: {len(valid_predictions)}")
print(f"- Predicciones nulas: {results_df['col2'].isna().sum()}")

if len(valid_predictions) > 0:
    print(f"- Promedio: {valid_predictions.mean():.4f}")
    print(f"- Mínimo: {valid_predictions.min():.4f}")
    print(f"- Máximo: {valid_predictions.max():.4f}") 