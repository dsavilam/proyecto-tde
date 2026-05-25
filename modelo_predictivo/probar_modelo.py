import joblib
import pandas as pd
import os
import sys

print("=======================================")
print("   PRUEBA DEL MODELO PREDICTIVO IA")
print("=======================================\n")

# 1. Cargar el modelo con ruta segura
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_estrato_bajo.pkl')
try:
    modelo_cargado = joblib.load(model_path)
    print("[OK] Modelo cargado exitosamente.\n")
except Exception as e:
    print(f"[ERROR] Error al cargar el modelo: {e}")
    sys.exit()

# Features exactas con las que fue entrenado
features = ['SECTOR_IES_BIN', 'SEXO_BIN', 'MODALIDAD_MATRICULA', 'UBICACION_CIUDAD']

# Si el usuario pasó un archivo .txt por argumento
if len(sys.argv) > 1:
    archivo_txt = sys.argv[1]
    # Limpiar el flag si el usuario puso "--archivo.txt" en lugar de "archivo.txt"
    if archivo_txt.startswith("--"):
        archivo_txt = archivo_txt[2:]
        
    ruta_txt = os.path.join(script_dir, archivo_txt)
    
    try:
        with open(ruta_txt, 'r', encoding='utf-8') as f:
            lineas = f.readlines()
        
        # Buscar la primera linea que no sea un comentario ni este vacia
        datos = None
        for linea in lineas:
            linea_limpia = linea.strip()
            if linea_limpia and not linea_limpia.startswith('#'):
                datos = linea_limpia
                break
                
        if datos:
            # Convertir string "1, 0, 1" a arreglo de enteros
            caracteristicas = [int(x.strip()) for x in datos.split(',')]
            
            print("-" * 50)
            print(f"LEYENDO ESTUDIANTE DESDE EL ARCHIVO: {archivo_txt}")
            print(f"Caracteristicas pasadas al modelo: {caracteristicas}")
            
            # Predecir
            df_estudiante = pd.DataFrame([caracteristicas], columns=features)
            prediccion = modelo_cargado.predict(df_estudiante)[0]
            
            # Interpretar
            resultado = ">> ESTRATO BAJO (1 o 2) - Poblacion Vulnerable" if prediccion == 1 else ">> ESTRATO MEDIO/ALTO"
            print(f"PREDICCION DEL MODELO: {resultado}")
            print("-" * 50)
        else:
            print(f"[ERROR] No se encontraron datos validos en {archivo_txt}.")
            
    except Exception as e:
        print(f"[ERROR] No se pudo procesar el archivo {archivo_txt}: {e}")
        
    sys.exit()

# 2. Si no hay argumentos, Definir una lista de estudiantes de prueba
# Tuplas formato: (Descripcion_humana, Arreglo_variables)
estudiantes_prueba = [
    (
        "Mujer en Ciudad / Sector Privado (Matricula)", 
        [1, 1, 1, 1] # Privado=1, Fem=1, Matr=1, Ciudad=1
    ),
    (
        "Hombre en Zona Rural / Sector Oficial (Sostenimiento)", 
        [0, 0, 0, 0] # Privado=0, Fem=0, Matr=0, Ciudad=0
    ),
    (
        "Mujer en Zona Rural / Sector Privado (Matricula)", 
        [1, 1, 1, 0] # Privado=1, Fem=1, Matr=1, Ciudad=0
    ),
    (
        "Hombre en Ciudad / Sector Oficial (Matricula)", 
        [0, 0, 1, 1] # Privado=0, Fem=0, Matr=1, Ciudad=1
    )
]

# 3. Iterar y predecir
for i, (descripcion, caracteristicas) in enumerate(estudiantes_prueba, 1):
    print("-" * 50)
    print(f"ESTUDIANTE {i}: {descripcion}")
    
    # Mostrar desglose humano de variables
    print(f"Caracteristicas pasadas al modelo: {caracteristicas}")
    
    # Predecir
    df_estudiante = pd.DataFrame([caracteristicas], columns=features)
    prediccion = modelo_cargado.predict(df_estudiante)[0]
    
    # Interpretar
    resultado = ">> ESTRATO BAJO (1 o 2) - Poblacion Vulnerable" if prediccion == 1 else ">> ESTRATO MEDIO/ALTO"
    print(f"PREDICCION DEL MODELO: {resultado}")
    
print("-" * 50)
print("\n¡Prueba finalizada! Puedes agregar más estudiantes en el código o usar un .txt para seguir probando.")
