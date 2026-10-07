# Nova application image (API + Streamlit UI). The LLM is NOT inside this image:
# point OLLAMA_BASE_URL at an Ollama server, or use NOVA_LLM_PROVIDER=openai_compatible.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NOVA_ENV=production \
    NOVA_DATA_DIR=/data \
    NOVA_LOG_DIR=/data/logs

WORKDIR /app

# libgomp is required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
# Whisper weights are downloaded on first use into the HF cache; persist it with a volume.
ENV HF_HOME=/data/hf-cache

COPY nova ./nova
COPY api ./api
COPY ui ./ui
COPY pages ./pages
COPY evaluation ./evaluation
COPY .streamlit ./.streamlit
COPY streamlit_app_pro.py pyproject.toml ./

RUN useradd --create-home --uid 10001 nova && mkdir -p /data && chown -R nova:nova /data /app
USER nova
VOLUME ["/data"]
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
