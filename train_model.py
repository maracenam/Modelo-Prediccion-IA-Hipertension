"""
train_model.py — Entrenamiento del modelo predictivo de Hipertensión Arterial (HTA)

Este script reproduce el proceso completo de entrenamiento del modelo:
  1. Carga y limpieza del dataset ENS 2016–2017
  2. Ingeniería de variables
  3. Comparación de 3 algoritmos (Regresión Logística, Árbol de Decisión, Random Forest)
  4. Selección del modelo con mayor sensibilidad
  5. Validación externa con datos del CESFAM Symon Ojeda
  6. Exportación del pipeline final como modelo_hta.pkl

Fuentes de datos (NO incluidas en el repositorio por privacidad):
  - ENS2017.csv           : Encuesta Nacional de Salud 2016–2017 (MINSAL)
  - Estratificación_final_SO_v1.csv : Registro clínico CESFAM Symon Ojeda (2022–2024)

Autor: Matias Aracena
Versión: 1.0.0
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURACIÓN DE RUTAS
# =====================================================================

ENS_PATH     = os.environ.get("ENS_PATH",     "data/ENS2017.csv")
CESFAM_PATH  = os.environ.get("CESFAM_PATH",  "data/Estratificación_final_SO_v1.csv")
MODEL_OUTPUT = os.environ.get("MODEL_OUTPUT", "modelo_hta.pkl")

# =====================================================================
# PASO 1: CARGA Y LIMPIEZA DEL DATASET ENS 2016–2017
# =====================================================================

def cargar_ens(path: str) -> pd.DataFrame:
    """
    Carga el dataset ENS, aplica limpieza y codificación de variables.

    Codificación de variables originales ENS:
        Sexo  : 1=Masculino, 2=Femenino
        HTA   : '1'=Hipertenso, '0'=Sano, ' '=Sin dato (excluir)
        ta3   : 1=Fumador actual, 2=Ex-fumador, 3=Nunca fumó, 4=Sin dato
        m7p2  : '1'=Consume alcohol, '2'=No consume, ' '=Sin dato
        di1   : 1=No tiene diabetes, 2=Sí tiene diabetes, 3=No sabe
        af1a  : 1=Sí tiene antecedentes familiares HTA, 2=No, -8888=Sin dato
        a4    : 1=Realiza actividad vigorosa, 2=No
        a7    : 1=Realiza actividad moderada, 2=No
        a13   : 1=Camina ≥10 min seguidos, 2=No
        IMC   : Valor numérico como string con coma decimal, ' '=Sin dato
    """
    print(f"[1/5] Cargando ENS desde '{path}'...")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    print(f"      {len(df)} filas cargadas, {df.shape[1]} columnas.")

    # --- Filtrar filas sin diagnóstico HTA ---
    df = df[df["HTA"].isin(["0", "1"])].copy()
    df["HTA"] = df["HTA"].astype(int)

    # --- IMC: convertir string con coma a float, descartar vacíos ---
    df["IMC"] = df["IMC"].astype(str).str.strip().str.replace(",", ".")
    df = df[df["IMC"] != ""].copy()
    df["IMC_Calculado"] = pd.to_numeric(df["IMC"], errors="coerce")
    df = df.dropna(subset=["IMC_Calculado"])

    # --- Sexo: 1=M→1, 2=F→0 ---
    df["Sexo"] = df["Sexo"].map({1: 1, 2: 0})

    # --- Tabaquismo: fumador activo (ta3==1) → 1, resto → 0 ---
    df["ta3"] = df["ta3"].apply(lambda x: 1 if x == 1 else 0)

    # --- Alcohol: '1'=consume → 1, '2'=no → 0, ' '=excluir ---
    df = df[df["m7p2"].astype(str).str.strip().isin(["1", "2"])].copy()
    df["m7p2"] = df["m7p2"].astype(str).str.strip().map({"1": 1, "2": 0})

    # --- Diabetes: di1==2 → 1 (tiene diabetes), resto → 0 ---
    df["di1"] = df["di1"].apply(lambda x: 1 if x == 2 else 0)

    # --- Antecedentes familiares HTA: af1a==1 → 1, resto → 0 ---
    df = df[df["af1a"].isin([1, 2])].copy()
    df["af1a"] = df["af1a"].map({1: 1, 2: 0})

    # --- Actividad física: activo si realiza al menos una de las 3 actividades ---
    # a4=vigorosa, a7=moderada, a13=camina (1=Sí, 2=No en ENS)
    df["Actividad_Fisica_Real"] = df[["a4", "a7", "a13"]].apply(
        lambda row: 300 if (row == 1).any() else 0, axis=1
    )

    columnas_finales = [
        "Edad", "Sexo", "IMC_Calculado",
        "ta3", "m7p2", "Actividad_Fisica_Real",
        "af1a", "di1", "HTA"
    ]
    df = df[columnas_finales].dropna()
    print(f"      Dataset limpio: {len(df)} filas | HTA positivos: {df['HTA'].sum()} ({df['HTA'].mean()*100:.1f}%)")
    return df


# =====================================================================
# PASO 2: ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# =====================================================================

def entrenar_y_comparar(df: pd.DataFrame) -> tuple:
    """
    Entrena 3 algoritmos con el mismo split 80/20 y retorna el mejor
    según sensibilidad (recall), priorizando minimizar falsos negativos.
    """
    print("\n[2/5] Entrenando modelos (split 80/20)...")

    X = df.drop(columns=["HTA"])
    y = df["HTA"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidatos = {
        "Regresión Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Árbol de Decisión": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
    }

    print(f"\n{'Algoritmo':<25} {'Exactitud':>10} {'Sensibilidad':>13} {'Precisión':>10}")
    print("-" * 62)

    resultados = {}
    for nombre, pipeline in candidatos.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        exactitud    = accuracy_score(y_test, y_pred)
        sensibilidad = recall_score(y_test, y_pred)
        precision    = precision_score(y_test, y_pred)

        resultados[nombre] = {
            "pipeline":     pipeline,
            "exactitud":    exactitud,
            "sensibilidad": sensibilidad,
            "precision":    precision,
        }
        print(f"{nombre:<25} {exactitud*100:>9.1f}% {sensibilidad*100:>12.1f}% {precision*100:>9.1f}%")

    # Seleccionar por mayor sensibilidad
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["sensibilidad"])
    mejor = resultados[mejor_nombre]
    print(f"\n✅ Modelo seleccionado: {mejor_nombre} (sensibilidad: {mejor['sensibilidad']*100:.1f}%)")
    print("   Criterio: minimizar falsos negativos (pacientes hipertensos no detectados)")

    return mejor["pipeline"], X_test, y_test


# =====================================================================
# PASO 3: VALIDACIÓN EXTERNA CON DATOS CESFAM SYMON OJEDA
# =====================================================================

def validar_cesfam(pipeline, path: str) -> None:
    """
    Valida el modelo con el dataset del CESFAM Symon Ojeda (2022–2024).
    Este dataset tiene variables distintas que se homologan al formato ENS.

    Homologación de variables CESFAM → modelo:
        AÑOS       → Edad
        SEXO       → Sexo (MASCULINO→1, FEMENINO→0)
        HTA        → variable objetivo (0/1)
        DIABETES   → di1
        TABAQUISMO → ta3
        CONSUMO ALCOHOL → m7p2
        OBESIDAD   → proxy de IMC alto (IMC estimado: obeso=32, normal=23)
        (sin columnas de actividad física ni antecedentes → se imputa con 0)
    """
    print(f"\n[3/5] Validación externa con CESFAM: '{path}'...")

    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
    print(f"      {len(df)} filas cargadas.")

    # Filtrar filas con HTA definido
    df = df[df["HTA"].isin([0, 1, 0.0, 1.0])].copy()
    df["HTA"] = df["HTA"].astype(int)

    # Sexo
    df["Sexo"] = df["SEXO"].astype(str).str.strip().str.upper().map(
        {"MASCULINO": 1, "FEMENINO": 0}
    )

    # Edad
    df["Edad"] = pd.to_numeric(df["AÑOS"], errors="coerce")

    # Diabetes: columna binaria 0/1
    df["di1"] = pd.to_numeric(df["DIABETES"], errors="coerce").fillna(0).astype(int)

    # Tabaquismo
    df["ta3"] = pd.to_numeric(df["TABAQUISMO"], errors="coerce").fillna(0).astype(int)

    # Alcohol
    df["m7p2"] = pd.to_numeric(df["CONSUMO ALCOHOL"], errors="coerce").fillna(0).astype(int)

    # IMC estimado a partir de columna OBESIDAD (proxy)
    # Sin IMC real: obeso → 32 (promedio obesidad grado I), no obeso → 23 (normopeso)
    df["OBESIDAD_NUM"] = pd.to_numeric(df["OBESIDAD"], errors="coerce").fillna(0)
    df["IMC_Calculado"] = df["OBESIDAD_NUM"].apply(lambda x: 32.0 if x == 1 else 23.0)

    # Actividad física y antecedentes no disponibles → imputar con 0
    df["Actividad_Fisica_Real"] = 0
    df["af1a"] = 0

    columnas_modelo = [
        "Edad", "Sexo", "IMC_Calculado",
        "ta3", "m7p2", "Actividad_Fisica_Real",
        "af1a", "di1"
    ]

    df_val = df[columnas_modelo + ["HTA"]].dropna()
    print(f"      Filas válidas para validación: {len(df_val)}")

    X_val = df_val[columnas_modelo]
    y_val = df_val["HTA"]

    y_pred = pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n      === REPORTE VALIDACIÓN EXTERNA (CESFAM) ===")
    print(f"      Exactitud  : {accuracy_score(y_val, y_pred)*100:.2f}%")
    print(f"      Sensibilidad: {recall_score(y_val, y_pred)*100:.2f}%")
    print(f"      Precisión  : {precision_score(y_val, y_pred)*100:.2f}%")
    print(f"\n      Matriz de confusión:")
    print(f"      Pacientes sanos correctamente clasificados : {cm[0][0]}")
    print(f"      Falsos positivos (alarma innecesaria)      : {cm[0][1]}")
    print(f"      Falsos negativos (enfermos no detectados)  : {cm[1][0]}")
    print(f"      Verdaderos positivos (HTA detectada)       : {cm[1][1]}")


# =====================================================================
# PASO 4: EXPORTAR MODELO
# =====================================================================

def exportar_modelo(pipeline, output_path: str) -> None:
    """Guarda el pipeline entrenado en formato pickle."""
    print(f"\n[4/5] Exportando modelo a '{output_path}'...")
    with open(output_path, "wb") as f:
        pickle.dump({"pipeline": pipeline}, f)
    print(f"      ✅ Modelo guardado correctamente.")


# =====================================================================
# PUNTO DE ENTRADA
# =====================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  HTA Predictor — Entrenamiento del Modelo")
    print("=" * 62)

    # Verificar que existan los archivos de datos
    for path, nombre in [(ENS_PATH, "ENS"), (CESFAM_PATH, "CESFAM")]:
        if not os.path.exists(path):
            print(f"\n❌ No se encontró el archivo {nombre}: '{path}'")
            print(f"   Configura la variable de entorno correspondiente o coloca")
            print(f"   el archivo en la ruta indicada.")
            sys.exit(1)

    df_ens       = cargar_ens(ENS_PATH)
    pipeline, X_test, y_test = entrenar_y_comparar(df_ens)

    if os.path.exists(CESFAM_PATH):
        validar_cesfam(pipeline, CESFAM_PATH)
    else:
        print("\n[3/5] ⚠️  Dataset CESFAM no encontrado. Se omite validación externa.")

    exportar_modelo(pipeline, MODEL_OUTPUT)

    print(f"\n[5/5] ✅ Pipeline completo. Modelo disponible en: '{MODEL_OUTPUT}'")
    print("=" * 62)
