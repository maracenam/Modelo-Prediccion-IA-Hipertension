# 📂 Datos de Entrenamiento

Los archivos de datos **no se incluyen en este repositorio** por razones de privacidad y licenciamiento.

## Fuentes utilizadas

### 1. ENS 2016–2017 (Encuesta Nacional de Salud)

| Campo | Detalle |
|-------|---------|
| Fuente | Ministerio de Salud de Chile (MINSAL) |
| Cobertura | Muestra nacional representativa de hogares chilenos |
| Formato | CSV separado por `;`, encoding UTF-8 |
| Nombre esperado | `ENS2017.csv` |
| Filas usadas | ~5.200 (tras limpieza) |

Puedes solicitar el dataset en: https://epi.minsal.cl/encuesta-nacional-de-salud/

---

### 2. Registro Clínico CESFAM Symon Ojeda (2022–2024)

| Campo | Detalle |
|-------|---------|
| Fuente | CESFAM Symon Ojeda, comuna de Conchalí |
| Cobertura | Pacientes estratificados por 51 patologías de interés epidemiológico |
| Formato | CSV separado por `;`, encoding Latin-1 |
| Nombre esperado | `Estratificación_final_SO_v1.csv` |
| Filas usadas | ~13.700 (tras limpieza) |
| Anonimización | Todos los datos fueron anonimizados antes de su uso |

Este dataset fue cedido por personal del CESFAM exclusivamente para uso académico.

---

## Cómo usar el script de entrenamiento

Una vez que dispongas de los archivos, colócalos en esta carpeta y ejecuta:

```bash
# Opción 1: archivos en carpeta data/
python train_model.py

# Opción 2: rutas personalizadas
ENS_PATH=ruta/ENS2017.csv CESFAM_PATH=ruta/cesfam.csv python train_model.py
```

El script genera `modelo_hta.pkl` en la raíz del proyecto.
