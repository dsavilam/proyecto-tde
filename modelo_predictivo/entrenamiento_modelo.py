import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- INICIANDO ENTRENAMIENTO DEL MODELO ---")
# 1. Cargar Vista Minable
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'dataset', 'vista_minable.csv')
df = pd.read_csv(csv_path)

# 2. Definir Target (Estrato Bajo = 1, Otros = 0)
df['TARGET_ESTRATO_BAJO'] = np.where(df['ESTRATO_CATEGORIA'] == 'Bajo', 1, 0)

# 3. Crear variables binarias colapsadas para evitar multicolinealidad
# SECTOR_IES: 1 para Privado, 0 para Oficial
df['SECTOR_IES_BIN'] = df['SECTOR IES_PRIVADO'] 

# SEXO: 1 para Femenino, 0 para Masculino
df['SEXO_BIN'] = df['SEXO AL NACER_Femenino']

# MODALIDAD: 1 para Matricula, 0 para Sostenimiento
df['MODALIDAD_MATRICULA'] = df['MODALIDAD DEL CREDITO_MATRICULA']

# UBICACION: 1 para Ciudad/Aglomeracion, 0 para Otros (Rural/Intermedio)
df['UBICACION_CIUDAD'] = df['CATEGORIA MUNICIPIO ORIGEN_CIUDADES Y AGLOMERACIONES']

# 4. Seleccionar Features Simplificadas
features = ['SECTOR_IES_BIN', 'SEXO_BIN', 'MODALIDAD_MATRICULA', 'UBICACION_CIUDAD']

X = df[features]
y = df['TARGET_ESTRATO_BAJO']

# 5. Dividir Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Entrenar Modelo (Mejorando hiperparametros para captar mejor el Estrato Bajo)
# Le damos un poco mas de peso a la clase 1 (Estrato Bajo) para priorizar su deteccion
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=6)
model.fit(X_train, y_train)

# 7. Evaluar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

real_prop = y_test.mean() * 100
pred_prop = np.mean(y_pred) * 100

print(f"\n--- ANÁLISIS DE POBLACIÓN (Set de Prueba) ---")
print(f"Porcentaje REAL de Estrato Bajo (Ground Truth): {real_prop:.2f}%")
print(f"Porcentaje PREDICHO por el modelo: {pred_prop:.2f}%")

print(f"\n--- RENDIMIENTO DEL MODELO ---")
print(f"Accuracy: {acc:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\n--- IMPORTANCIA DE VARIABLES ---")
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

# 8. Guardar Modelo
joblib.dump(model, os.path.join(script_dir, 'modelo_estrato_bajo.pkl'))
print("\n[OK] Modelo guardado exitosamente en 'modelo_estrato_bajo.pkl'")
