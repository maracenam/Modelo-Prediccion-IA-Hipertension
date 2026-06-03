"""
HTA Predictor Bot - Telegram Chatbot
Predice el riesgo de hipertensión arterial a partir de datos simples del usuario.

Autor: Matias Aracena
Versión: 1.0.0
"""

import os
import sys
import threading
import requests
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
from predictor import HypertensionPredictor

# =====================================================================
# CONFIGURACIÓN
# Carga el token desde variable de entorno para no exponerlo en el código.
# Configúralo con: export TELEGRAM_TOKEN="tu_token_aqui"
# =====================================================================
TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    print("❌ Error: la variable de entorno TELEGRAM_TOKEN no está definida.")
    sys.exit(1)

PORT = int(os.environ.get("PORT", 5050))

app = Flask(__name__)

# Carga del modelo al iniciar la aplicación
try:
    print("⏳ Cargando modelo de IA...")
    modelo = HypertensionPredictor()
    print("✅ Modelo listo.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    sys.exit(1)

# Almacén de sesiones activas por chat_id
sessions = {}


# =====================================================================
# FUNCIONES DE COMUNICACIÓN CON TELEGRAM
# =====================================================================

def enviar_mensaje(chat_id: int, text: str) -> None:
    """Envía un mensaje de texto al usuario en Telegram."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[ERROR] No se pudo enviar mensaje a {chat_id}: {e}")


# =====================================================================
# MÁQUINA DE ESTADOS: PROCESAMIENTO DE MENSAJES
# =====================================================================

def procesar_mensaje(chat_id: int, text: str) -> None:
    """
    Procesa cada mensaje del usuario según el paso actual de su sesión.
    Implementa una máquina de estados para guiar el flujo de preguntas.
    """
    try:
        # --- INICIO DE SESIÓN ---
        if text == "/start":
            sessions[chat_id] = {
                "step": "consentimiento",
                "data": {"imc_directo": None, "peso": None, "talla": None},
            }
            enviar_mensaje(
                chat_id,
                (
                    "👋 *Hola.* Soy un asistente que te ayudará a estimar tu posible riesgo de hipertensión.\n\n"
                    "Antes de continuar, debes saber lo siguiente:\n"
                    "🔹 Esta herramienta es *informativa*, orientada a prevención.\n"
                    "🔹 *No reemplaza* una consulta médica.\n"
                    "🔹 Solo se procesarán los datos que tú decidas entregar.\n\n"
                    "📄 *Consentimiento:* ¿Autorizas procesar tus datos? (SI/NO)"
                ),
            )
            return

        # Verificar sesión activa
        if chat_id not in sessions:
            enviar_mensaje(chat_id, "⚠️ Sesión expirada. Escribe /start para comenzar.")
            return

        session = sessions[chat_id]
        step = session["step"]

        # --- FLUJO DE PASOS ---

        if step == "consentimiento":
            if text.upper() in ["SI", "S"]:
                session["step"] = "edad"
                enviar_mensaje(chat_id, "✅ Gracias.\n\n1️⃣ *¿Cuál es tu edad?* (Ej: 45)")
            elif text.upper() in ["NO", "N"]:
                del sessions[chat_id]
                enviar_mensaje(chat_id, "Entiendo 👍. Para proteger tu privacidad, cerraré la sesión.")
            else:
                enviar_mensaje(chat_id, "⚠️ Por favor responde *SI* o *NO*.")

        elif step == "edad":
            if text.isdigit() and 15 <= int(text) <= 110:
                session["data"]["edad"] = int(text)
                session["step"] = "sexo"
                enviar_mensaje(chat_id, "2️⃣ *Sexo biológico (M/F):*")
            else:
                enviar_mensaje(chat_id, "❌ Edad inválida. Ingresa un valor entre 15 y 110.")

        elif step == "sexo":
            if text.upper() in ["M", "F", "MASCULINO", "FEMENINO"]:
                session["data"]["sexo"] = "M" if text.upper().startswith("M") else "F"
                session["step"] = "consulta_imc"
                enviar_mensaje(chat_id, "3️⃣ *¿Conoces tu IMC?* (SI/NO)")
            else:
                enviar_mensaje(chat_id, "❌ Ingresa *M* (Masculino) o *F* (Femenino).")

        elif step == "consulta_imc":
            if text.upper() in ["SI", "S"]:
                session["step"] = "ingreso_imc"
                enviar_mensaje(chat_id, "🔢 *Ingresa tu IMC* (Ej: 25.4):")
            elif text.upper() in ["NO", "N"]:
                session["step"] = "peso"
                enviar_mensaje(chat_id, "⚖️ *Ingresa tu peso en kg* (Ej: 70.5):")
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

        elif step == "ingreso_imc":
            try:
                val = float(text.replace(",", "."))
                if not (10 <= val <= 60):
                    raise ValueError
                session["data"]["imc_directo"] = val
                session["step"] = "tabaco"
                enviar_mensaje(chat_id, "📋 *Hábitos*\n\n¿Fumas actualmente? (SI/NO)")
            except ValueError:
                enviar_mensaje(chat_id, "❌ IMC inválido. Debe estar entre 10 y 60 (Ej: 25.4).")

        elif step == "peso":
            try:
                val = float(text.replace(",", "."))
                if not (30 <= val <= 200):
                    raise ValueError
                session["data"]["peso"] = val
                session["step"] = "talla"
                enviar_mensaje(chat_id, "📏 *Ingresa tu estatura en cm* (Ej: 175):")
            except ValueError:
                enviar_mensaje(chat_id, "❌ Peso inválido. Ingresa un valor entre 30 y 200 kg.")

        elif step == "talla":
            try:
                val = float(text)
                if not (100 <= val <= 230):
                    raise ValueError
                session["data"]["talla"] = val
                session["step"] = "tabaco"
                enviar_mensaje(chat_id, "📋 *Hábitos*\n\n¿Fumas actualmente? (SI/NO)")
            except ValueError:
                enviar_mensaje(chat_id, "❌ Estatura inválida. Ingresa un valor entre 100 y 230 cm.")

        elif step == "tabaco":
            if text.upper() in ["SI", "S", "NO", "N"]:
                session["data"]["tabaco"] = "S" if "S" in text.upper() else "N"
                session["step"] = "alcohol"
                enviar_mensaje(chat_id, "🍺 *¿Consumes alcohol habitualmente?* (SI/NO)")
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

        elif step == "alcohol":
            if text.upper() in ["SI", "S", "NO", "N"]:
                session["data"]["alcohol"] = "S" if "S" in text.upper() else "N"
                session["step"] = "diabetes"
                enviar_mensaje(chat_id, "🩸 *¿Tienes diabetes diagnosticada?* (SI/NO)")
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

        elif step == "diabetes":
            if text.upper() in ["SI", "S", "NO", "N"]:
                session["data"]["diabetes"] = "S" if "S" in text.upper() else "N"
                session["step"] = "antecedentes"
                enviar_mensaje(chat_id, "🧬 *¿Tienes antecedentes familiares de hipertensión?* (SI/NO)")
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

        elif step == "antecedentes":
            if text.upper() in ["SI", "S", "NO", "N"]:
                session["data"]["antecedentes"] = "S" if "S" in text.upper() else "N"
                session["step"] = "actividad"
                enviar_mensaje(chat_id, "🏃 *¿Realizas actividad física regular?* (SI/NO)")
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

        elif step == "actividad":
            if text.upper() in ["SI", "S", "NO", "N"]:
                session["data"]["actividad"] = "S" if "S" in text.upper() else "N"

                # --- PREDICCIÓN FINAL ---
                try:
                    res = modelo.predecir(session["data"])
                    msg = (
                        f"{res['titulo']}\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"{res['mensaje']}\n\n"
                        f"👤 *Tu IMC:* {res['imc']}\n\n"
                        f"{res['recomendacion']}"
                    )
                    enviar_mensaje(chat_id, msg)
                except Exception as e:
                    print(f"[ERROR] Fallo en predicción: {e}")
                    enviar_mensaje(chat_id, "⚠️ Ocurrió un error técnico. Por favor intenta más tarde.")

                del sessions[chat_id]
            else:
                enviar_mensaje(chat_id, "⚠️ Responde *SI* o *NO*.")

    except Exception as e:
        print(f"[ERROR] Excepción en procesamiento de mensaje: {e}")


# =====================================================================
# WEBHOOK - ENDPOINT FLASK
# =====================================================================

@app.route("/", methods=["POST"])
def webhook():
    """Recibe actualizaciones de Telegram y las procesa en un hilo separado."""
    try:
        data = request.json
        if data and "message" in data:
            chat_id = data["message"]["chat"]["id"]
            text = data["message"].get("text", "").strip()
            threading.Thread(target=procesar_mensaje, args=(chat_id, text)).start()
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[ERROR] Webhook: {e}")
        return jsonify({"status": "error"}), 500


# =====================================================================
# PUNTO DE ENTRADA
# =====================================================================

if __name__ == "__main__":
    conf.get_default().monitor_thread = False

    # Limpiar túneles ngrok anteriores
    for tunnel in ngrok.get_tunnels():
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass

    try:
        public_url = ngrok.connect(f"127.0.0.1:{PORT}").public_url
        print(f"\n✅ Bot online en: {public_url}")
        requests.get(
            f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={public_url}",
            timeout=10,
        )
        print("🚀 Esperando usuarios...\n")
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error al iniciar: {e}")
