# ──────────────────────────────────────────────────────────────────────
# Imagen base: Python 3.12 slim (ligero, ~120 MB)
# ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ──────────────────────────────────────────────────────────────────────
# Variables de entorno
# ──────────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ──────────────────────────────────────────────────────────────────────
# Directorio de trabajo dentro del contenedor
# ──────────────────────────────────────────────────────────────────────
WORKDIR /app

# ──────────────────────────────────────────────────────────────────────
# Instalar dependencias del sistema (mínimas)
# ──────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────
# Instalar dependencias Python (capa cacheable)
# ──────────────────────────────────────────────────────────────────────
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt

# ──────────────────────────────────────────────────────────────────────
# Copiar el código de la aplicación
# ──────────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY models/ ./models/

# ──────────────────────────────────────────────────────────────────────
# Exponer el puerto y arrancar uvicorn
# ──────────────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]