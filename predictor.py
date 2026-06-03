"""
HypertensionPredictor - Módulo de predicción de riesgo de hipertensión arterial.

Carga el pipeline de ML entrenado (modelo_hta.pkl) y expone el método `predecir`
que recibe los datos del usuario y retorna un resultado cualitativo.

Autor: Matias Aracena
Versión: 1.0.0
"""

import os
import sys
import pickle
import pandas as pd


class HypertensionPredictor:
    """
    Encapsula el pipeline de Regresión Logística entrenado para predecir
    el riesgo de hipertensión arterial a partir de datos simples del usuario.

    Variables de entrada del modelo:
        - Edad                 : int   — edad en años
        - Sexo                 : int   — 1 = Masculino, 0 = Femenino
        - IMC_Calculado        : float — índice de masa corporal
        - ta3                  : int   — tabaquismo activo (1 = Sí)
        - m7p2                 : int   — consumo habitual de alcohol (1 = Sí)
        - Actividad_Fisica_Real: int   — minutos semanales estimados (300 = activo, 0 = sedentario)
        - af1a                 : int   — antecedentes familiares de HTA (1 = Sí)
        - di1                  : int   — diagnóstico de diabetes (1 = Sí)
    """

    # Umbral de decisión: probabilidad >= 0.40 activa la alerta de riesgo.
    UMBRAL_ALERTA = 0.40

    def __init__(self, model_path: str = "modelo_hta.pkl"):
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado en '{model_path}'")
            sys.exit(1)

        with open(model_path, "rb") as f:
            data_pkl = pickle.load(f)
            self.pipeline = data_pkl["pipeline"]

        print(f"   Modelo cargado desde '{model_path}'")

    # ------------------------------------------------------------------
    # PREDICCIÓN
    # ------------------------------------------------------------------

    def predecir(self, data: dict) -> dict:
        """
        Recibe los datos recopilados por el chatbot y retorna un resultado
        cualitativo (sin valores numéricos de probabilidad).

        Args:
            data (dict): Diccionario con las claves:
                edad, sexo, tabaco, alcohol, diabetes, antecedentes, actividad,
                imc_directo (opcional), peso (opcional), talla (opcional).

        Returns:
            dict: Contiene 'titulo', 'mensaje', 'recomendacion' e 'imc'.
        """
        imc = self._calcular_imc(data)
        df = self._construir_dataframe(data, imc)
        prob = self.pipeline.predict_proba(df)[0][1]
        return self._generar_respuesta(prob, imc)

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # ------------------------------------------------------------------

    def _calcular_imc(self, data: dict) -> float:
        """Calcula el IMC a partir del peso/talla o usa el valor directo ingresado."""
        if data.get("imc_directo") is not None:
            return data["imc_directo"]
        talla_m = data["talla"] / 100.0
        return data["peso"] / (talla_m ** 2)

    def _construir_dataframe(self, data: dict, imc: float) -> pd.DataFrame:
        """Construye el DataFrame de entrada para el pipeline."""
        return pd.DataFrame({
            "Edad":                 [data["edad"]],
            "Sexo":                 [1 if data["sexo"] == "M" else 0],
            "IMC_Calculado":        [imc],
            "ta3":                  [1 if data["tabaco"] == "S" else 0],
            "m7p2":                 [1 if data["alcohol"] == "S" else 0],
            "Actividad_Fisica_Real":[300 if data["actividad"] == "S" else 0],
            "af1a":                 [1 if data["antecedentes"] == "S" else 0],
            "di1":                  [1 if data["diabetes"] == "S" else 0],
        })

    def _generar_respuesta(self, prob: float, imc: float) -> dict:
        """Genera el mensaje de salida según si supera o no el umbral de alerta."""
        links = (
            "\n\nMás información sobre el sistema →\n🔗 [Ver documento](https://bit.ly/IA-HTA)\n\n"
            "Tu información está protegida: conexión cifrada, validación de datos y accesos controlados.\n"
            "🔗 [Política de privacidad](https://bit.ly/IA-HTA)\n\n"
            "Documentación del modelo (propósito, datos, limitaciones) →\n"
            "🔗 [Ver documentación](https://bit.ly/IA-HTA)\n\n"
            "Evaluación de equidad por subgrupos (edad y sexo) →\n"
            "🔗 [Ver resultados](https://bit.ly/IA-HTA)"
        )

        if prob >= self.UMBRAL_ALERTA:
            return {
                "titulo": "⚠️ POSIBLE HIPERTENSIÓN DETECTADA",
                "mensaje": (
                    "Según los factores de riesgo ingresados, existe una probabilidad "
                    "significativa de hipertensión."
                ),
                "recomendacion": (
                    "🚨 *ACCIÓN RECOMENDADA:*\n"
                    "Te sugerimos acudir a tu CESFAM o centro médico más cercano para "
                    "una medición de presión arterial profesional. "
                    "La detección temprana es clave."
                    + links
                ),
                "imc": round(imc, 1),
            }
        else:
            return {
                "titulo": "✅ RIESGO BAJO (NORMOTENSIÓN)",
                "mensaje": "Tus indicadores actuales se mantienen bajo el umbral de riesgo crítico.",
                "recomendacion": (
                    "🌿 *CONSEJOS DE VIDA SALUDABLE:*\n"
                    "• Mantén una dieta baja en sodio (sal).\n"
                    "• Realiza al menos 30 min de caminata al día.\n"
                    "• Evita el tabaco y el consumo excesivo de alcohol.\n"
                    "• Realízate un chequeo preventivo una vez al año."
                    + links
                ),
                "imc": round(imc, 1),
            }
