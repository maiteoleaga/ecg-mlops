# ECG Heartbeat Classification — MLOps

Clasificación supervisada de latidos de electrocardiograma (ECG) en 5 categorías clínicas mediante una red neuronal **InceptionTime 1D** entrenada sobre el dataset MIT-BIH Arrhythmia.

Este proyecto es la práctica final de la asignatura **MLOps** del Máster en Deep Learning de la Universidad Politécnica de Madrid. Se centra en aplicar metodologías, buenas prácticas y herramientas de MLOps sobre un modelo previamente desarrollado en la asignatura *Deep Learning para Series Temporales*.

## Autor

- **Maite Oleaga Laespada** 

Máster en Deep Learning — Universidad Politécnica de Madrid

## Problema

Clasificar latidos cardíacos individuales (series temporales de 187 puntos) en una de 5 clases:

| Etiqueta | Clase | Descripción |
|----------|-------|-------------|
| 0 | N | Latido normal |
| 1 | S | Supraventricular |
| 2 | V | Ventricular |
| 3 | F | Fusión |
| 4 | Q | Desconocido / no clasificable |

El dataset presenta un fuerte desbalance: la clase N representa ~83 % de las muestras. El proyecto utiliza F1-macro como métrica principal en lugar de accuracy.

## Modelo

**InceptionTime 1D** (sin class weights), arquitectura compuesta por 6 módulos Inception con convoluciones paralelas de múltiples tamaños de kernel + Global Average Pooling + capa lineal final. Entrenada con **fastai** sobre **PyTorch**.

Métricas en test:

| Métrica | Valor |
|---------|------:|
| Accuracy | 0.98 |
| F1-weighted | 0.98 |
| F1-macro | 0.94 |

## Estructura del proyecto

```
ecg-mlops/
├── .gitignore
├── README.md
├── requirements.txt
├── data/                # ignorado por git — descargar con scripts/download_data.py
│   ├── raw/
│   └── processed/
├── models/              # ignorado por git — descargar desde W&B
├── notebooks/
│   └── main.ipynb       # notebook original con el EDA y experimentos
├── src/                 # código de entrenamiento
├── api/                 # servicio FastAPI
├── tests/               # tests pytest
└── scripts/             # utilidades (descarga de datos, etc.)
```

## Requisitos

- macOS o Linux
- Python **3.12** (no compatible con 3.13)
- Git

## Instalación (entorno local de desarrollo)

```bash
# 1. Clonar el repositorio
git clone <url_del_repo>
cd ecg-mlops

# 2. Crear y activar el entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

## Datos

El proyecto utiliza el **MIT-BIH Arrhythmia Database** en formato CSV, disponible en Kaggle:

[ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

### Cómo obtenerlos

1. Crear una cuenta gratuita en [Kaggle](https://www.kaggle.com/).
2. Descargar el dataset desde el enlace de arriba.
3. Descomprimir el archivo `heartbeat.zip`.
4. Copiar los archivos `mitbih_train.csv` y `mitbih_test.csv` a la carpeta `data/raw/`

Estructura final esperada:
```
data/raw/
├── mitbih_train.csv   (~86 MB)
└── mitbih_test.csv    (~21 MB)
```

> Los archivos de datos **no se incluyen en el repositorio** (`data/` está en `.gitignore`). Cada usuario debe descargarlos por su cuenta siguiendo los pasos anteriores.


## Enlaces

- **Repositorio GitHub:** *pendiente*
- **Proyecto en Weights & Biases:** *pendiente*
- **Endpoint en producción:** *pendiente*

Estructura final esperada:
```
data/raw/
├── mitbih_train.csv   (~86 MB)
└── mitbih_test.csv    (~21 MB)
```

> Los archivos de datos **no se incluyen en el repositorio** (`data/` está en `.gitignore`). Cada usuario debe descargarlos por su cuenta siguiendo los pasos anteriores.