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


## Estructura del proyecto

```
ecg-mlops/
├── .github/workflows/     # GitHub Actions (CI)
├── api/                   # Servicio FastAPI
├── config/                # config.yaml (hiperparámetros, W&B)
├── data/                  # Datos (ignorado por git)
├── models/                # Modelo entrenado
├── notebooks/             # EDA y entrenamiento en Colab
├── src/                   # Código modular (data, model, train, evaluate)
├── tests/                 # Tests con pytest
├── Dockerfile
├── pytest.ini
├── requirements.txt       # Dependencias de desarrollo
└── requirements-api.txt   # Dependencias mínimas para la API
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

## Entrenamiento del modelo 
El entrenamiento se ejecuta con W&B activado para trackear métricas, hiperparámetros y artifacts del modelo. El entrenamiento final se ha realizado en Google Colab con GPU T4. El notebook `notebooks/train_colab.ipynb` clona el repositorio, lanza el entrenamiento y sube el modelo a Weights & Biases. Una vez completado, el modelo se descarga desde W&B a `models/inception_time.pt` para servirlo localmente. 

Para ejecutar el entrenamiento se usa el comando: 

```bash
python -m src.train --wandb --name nombre-del-run
```
### Hiperparámetros
Los parámetros se pueden editar en el archivo `config/config.yaml`, esto permite no modifcar los scripts y hacer diferentes experimentos con el mismo modelo. 

### Resultados obtenidos

**InceptionTime 1D** (sin class weights), arquitectura compuesta por 6 módulos Inception con convoluciones paralelas de múltiples tamaños de kernel + Global Average Pooling + capa lineal final. Entrenada con  **PyTorch**.

Métricas en test:

| Métrica | Valor |
|---------|------:|
| Accuracy | 0.99 |
| F1-weighted | 0.99 |
| F1-macro | 0.82 |

## API y uso

Con la API se puede probar el modelo entrenado. 

### Arrancar la API en local

```bash
uvicorn api.main:app --reload --port 8000
```

Se accede en `http://localhost:8000`.

### Endpoint principal: `POST /predict`

Recibe una señal de **187 puntos** y devuelve la clase predicha junto con las probabilidades por clase.

**Ejemplo de body:**

```json
{
  "signal": [1.0, 0.821, 0.221, 0.091, ..., 0.0, 0.0]
}
```

> El array `signal` debe contener exactamente 187 valores numéricos.

### Probar la API

La forma más sencilla de probar la API es a través de Swagger UI, accesible en /docs. Desde ahí puedes desplegar POST /predict, pulsar "Try it out" y enviar una señal de ejemplo directamente desde el navegador.


### Endpoint en producción

El endpoint está desplegado en Render.com en su versión gratuita. 

[https://ecg-mlops.onrender.com/docs](https://ecg-mlops.onrender.com/docs)

> ⚠️ Nota sobre Render Free: el servicio se duerme tras 15 min sin uso. La primera petición tras un periodo de inactividad puede tardar 30-60 segundos mientras Render reactiva el contenedor. Las siguientes peticiones serán rápidas.

## Tests

El proyecto incluye 14 test con pytest, organizados en 3 archivos: 

| Archivo | Qué verifica |
|---|---|
|`test_data.py`| Carga de CSVs, shapes, splits, dataloaders |
|`test_model.py`| Forward del modelo, parámetros, softmax |
|`test_api.py`| Endpoints, validaciones de Pydantic |

### Ejecutar todos los tests en local
Todos (necesita CSVs en data/raw/ y modelo en models/)
```bash
pytest -v
```

### Ejecutar solo los tests sin dependencias externas (modo CI)
Solo los que no requieren datos ni modelo (los que corre el CI)
```bash
pytest -v -m "not requires_data and not requires_model"
```

El CI ejecuta únicamente los tests que no requieren datos ni modelo, ya que los CSVs (data/) están excluidos del repositorio y el modelo se versiona aparte. Esta separación se gestiona mediante los marcadores `@pytest.mark.requires_data` y `@pytest.mark.requires_model` definidos en pytest.ini.

## Docker

La API se distribuye como una imagen Docker autocontenida que incluye Python, las dependencias y el modelo entrenado, lo que garantiza un despliegue reproducible en cualquier entorno.

### Estructura de la imagen

- **Imagen base**: `python:3.12-slim` (versión fijada y ligera).
- **Dependencias**: el proyecto incluye dos archivos de requirements. `requirements.txt` para desarrollo (con jupyter, wandb, pytest...) y `requirements-api.txt` con solo las librerías esenciales para servir la API. Este último es el que utiliza el Dockerfile, lo que reduce significativamente el tamaño de la imagen.
- **`.dockerignore`**: excluye del build todos los archivos que no se deben copiar al contenedor (`data/`, `notebooks/`, `tests/`, `.venv/`, etc.).
- **Modelo embebido**: el modelo entrenado (`models/inception_time.pt`) se incluye dentro de la imagen para que el contenedor sea autocontenido. La alternativa sería descargarlo desde W&B en runtime, lo cual exigiría exponer la API key de W&B en el contenedor.

### Construir la imagen

```bash
docker build -t ecg-mlops:latest .
```

### Arrancar el contenedor

```bash
docker run -d --name ecg-api -p 8000:8000 ecg-mlops:latest
```

La API queda accesible en `http://localhost:8000`.

## CI/CD
El proyecto cuenta con un pipeline de Integración Continua (CI) y Despliegue Continuo (CD). Ambos procesos son independientes y, aunque el CI puede existir sin CD, desplegar de forma automática sin haber validado antes los cambios no es una práctica recomendable; por eso aquí van encadenados.

### Integración Continua (CI)
El proyecto utiliza GitHub Actions como herramienta de CI. El workflow está definido en `.github/workflows/CI.yaml` y se ejecuta automáticamente con cada `push` a la rama `main`. En él se validan los tests y se comprueba que la imagen Docker se construye correctamente.

### Despliegue Continuo (CD)
Cada cambio aceptado en `main` se despliega automáticamente en producción. Los detalles del servicio desplegado y la URL pública se encuentran en la sección [API y uso](#api-y-uso).

## Enlaces

- [Repositorio GitHub](https://github.com/maiteoleaga/ecg-mlops)
- [Report en W&B](https://wandb.ai/maiteol/ecg-mlops/reports/ECG-Heartbeat-Classification-Comparativa-de-Learning-Rate--VmlldzoxNzE0OTMyOA?accessToken=8ycjgc7yzu691of5mb9su7yio00yhenvednmr5bk1nsvrm76o5zn0orvphk2t8el)
- [Endpoint en producción](https://ecg-mlops.onrender.com)
- [Swagger UI del endpoint](https://ecg-mlops.onrender.com/docs)




