# 🫀 HTA Predictor Bot

> Chatbot de Telegram con IA para estimar el riesgo de **hipertensión arterial** en adultos chilenos.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)](https://flask.palletsprojects.com)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-2CA5E0?logo=telegram)](https://core.telegram.org/bots)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Descripción

La hipertensión arterial afecta al **36% de la población adulta en Chile** y suele detectarse tardíamente por su carácter silencioso. Este proyecto desarrolla un modelo de Machine Learning integrado en un chatbot de Telegram que permite estimar el riesgo de HTA a partir de información simple y accesible, **sin necesidad de exámenes clínicos**.

El sistema guía al usuario a través de una conversación estructurada y entrega un resultado preventivo cualitativo (no un diagnóstico médico), promoviendo la consulta temprana con profesionales de salud.

---

## 🧠 Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Modelo ML | Regresión Logística (`scikit-learn`) |
| Backend API | Python + Flask |
| Tunnel público | ngrok |
| Interfaz de usuario | Telegram Bot API |
| Datos de entrenamiento | ENS 2016–2017 + CESFAM Symon Ojeda (2022–2024) |

---

## 🏗️ Arquitectura

```
Usuario (Telegram)
      │
      ▼
 Telegram API
      │  webhook POST
      ▼
  Flask App  ──►  Máquina de estados (sesiones por chat_id)
      │
      ▼
HypertensionPredictor
      │
      ▼
 Pipeline sklearn (.pkl)
      │  predict_proba()
      ▼
Resultado cualitativo  ──►  Telegram API  ──►  Usuario
```

---

## 🤖 Modelo de Machine Learning

Se evaluaron tres algoritmos supervisados sobre el mismo dataset (ENS 2016–2017, n=5.460):

| Algoritmo | Exactitud | Sensibilidad | Precisión |
|-----------|-----------|-------------|-----------|
| **Regresión Logística** ✅ | 80.5% | **69.5%** | 73.7% |
| Árbol de Decisión | 79.6% | 67.0% | 73.0% |
| Random Forest | 76.2% | 63.8% | 67.2% |

**Criterio de selección:** se priorizó la **sensibilidad (recall)** para minimizar falsos negativos, dado que el costo de no detectar un hipertenso es significativamente mayor que el de generar una falsa alarma.

**Validación externa** con datos del CESFAM Symon Ojeda (n=13.700 pacientes):

| Métrica | Resultado |
|---------|-----------|
| Exactitud | 74.98% |
| Sensibilidad | 58.55% |
| Precisión | 77.06% |
| Verdaderos positivos | 3.430 |
| Falsos negativos | 2.428 |

> **Nota sobre el umbral:** las métricas anteriores usan el umbral estándar de 0,50. En el chatbot se usa un umbral de **0,40** para ser más conservador (priorizar detectar casos), por lo que en uso real la sensibilidad es mayor y la precisión algo menor que lo reportado.

### Variables de entrada

| Variable | Descripción | Tipo |
|----------|-------------|------|
| Edad | Edad del usuario | Numérico |
| Sexo | Sexo biológico | Binario (M/F) |
| IMC | Índice de masa corporal | Numérico |
| Tabaquismo | Fumador activo | Binario |
| Alcohol | Consumo habitual | Binario |
| Actividad física | Actividad regular | Binario |
| Antecedentes familiares | Historia familiar de HTA | Binario |
| Diabetes | Diagnóstico previo | Binario |

---

## 🔍 Limitaciones

- **Validación externa con datos incompletos.** El registro del CESFAM no incluye IMC exacto (se estimó a partir de la marca de obesidad: 32 si es obeso, 23 si no), ni actividad física ni antecedentes familiares (se imputaron como 0). Esto explica buena parte de la baja de sensibilidad (69,5% → 58,6%).
- **Datos autorreportados.** Las variables de entrada las declara el usuario, sin verificación clínica.
- **Población.** El modelo se entrenó con población chilena adulta; no se ha evaluado en otros países.
- **Variables limitadas.** Solo 8 factores de riesgo; no incluye, por ejemplo, consumo de sal ni presión medida.
- **Uso académico de los datos.** El dataset del CESFAM fue cedido exclusivamente para fines académicos.

---

## 📸 Demo

<!-- Agrega aquí una captura o GIF del bot funcionando, por ejemplo: ![Demo del bot](docs/demo.gif) -->

---

## 💬 Flujo del chatbot

```
/start
  └─► Consentimiento
        └─► Edad → Sexo
              └─► ¿Conoce su IMC?
                    ├─► Sí → Ingreso directo de IMC
                    └─► No → Peso + Estatura → Cálculo automático
                          └─► Tabaco → Alcohol → Diabetes
                                └─► Antecedentes → Actividad física
                                      └─► Predicción + Resultado
```

**Posibles resultados:**
- ⚠️ **Posible HTA detectada** → Se recomienda acudir a CESFAM u otro centro de salud.
- ✅ **Riesgo bajo (normotensión)** → Se entregan consejos de vida saludable.

> El resultado es siempre cualitativo. El bot **nunca** entrega valores numéricos de probabilidad.

---

## ⚖️ IA Responsable

El proyecto incorpora explícitamente principios de IA responsable:

| Principio | Implementación |
|-----------|---------------|
| **Consentimiento** | El bot solicita autorización explícita antes de procesar cualquier dato |
| **Transparencia** | Se informa el propósito, limitaciones y que no reemplaza un diagnóstico médico |
| **Privacidad** | Solo se procesan datos entregados voluntariamente; no se almacena información sensible |
| **Equidad** | Se usaron dos fuentes de datos para reducir sesgos demográficos |
| **Responsabilidad** | Uso informativo, no clínico; se especifican los responsables del sistema |
| **Confiabilidad** | Comparación de múltiples algoritmos y validación con dataset externo |

---

## 🚀 Instalación y uso

### Prerrequisitos
- Python 3.10+
- Cuenta en [ngrok](https://ngrok.com/) (gratuita)
- Bot de Telegram creado con [@BotFather](https://t.me/botfather)

### 1. Clonar el repositorio

```bash
git clone https://github.com/maracenam/Modelo-Prediccion-IA-Hipertension.git
cd Modelo-Prediccion-IA-Hipertension
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env y agrega tu TELEGRAM_TOKEN
```

### 4. Entrenar el modelo (o usar uno existente)

Si tienes acceso a los datos fuente, entrena ejecutando:

```bash
# Coloca los CSVs en data/ y ejecuta:
python train_model.py

# O con rutas personalizadas:
ENS_PATH=ruta/ENS2017.csv CESFAM_PATH=ruta/cesfam.csv python train_model.py
```

Esto genera `modelo_hta.pkl` en la raíz. Ver [`data/README.md`](data/README.md) para detalles de las fuentes.

### 5. Ejecutar

```bash
export TELEGRAM_TOKEN="tu_token_aqui"   # macOS/Linux
# set TELEGRAM_TOKEN=tu_token_aqui      # Windows

python app.py
```

---

## 📁 Estructura del proyecto

```
predictor-hipertension-ia/
├── app.py              # Servidor Flask + webhook + máquina de estados
├── predictor.py        # Clase HypertensionPredictor (carga modelo y predice)
├── train_model.py      # Script de entrenamiento (reproduce el modelo desde cero)
├── modelo_hta.pkl      # Pipeline entrenado (generado por train_model.py, no en repo)
├── requirements.txt    # Dependencias Python
├── .env.example        # Plantilla de variables de entorno
├── data/
│   └── README.md       # Instrucciones para obtener los datos de entrenamiento
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Contexto del problema

- **36%** de la población adulta chilena tiene hipertensión (supera el promedio global).
- **73%** de los mayores de 65 años están afectados.
- La detección suele ocurrir de forma incidental o cuando ya hay complicaciones (infartos, ACV, insuficiencia renal).
- Este proyecto apunta a una **intervención temprana y proactiva**, aprovechando canales digitales de uso cotidiano.

---

## 👤 Autor

**Matias Aracena**  
Estudiante de Ingeniería · Universidad Diego Portales  
Asignatura: Inteligencia Artificial · Profesor: Claudio Fuentes

---

## ⚠️ Disclaimer

Esta herramienta es de uso **informativo y preventivo**. No reemplaza una consulta médica profesional. Ante cualquier sospecha de hipertensión, se recomienda acudir a un centro de salud.
