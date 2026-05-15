FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Install runtime dependencies for lightweight CPU servers
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
  fastapi \
  uvicorn[standard] \
  transformers

# Force CPU-only PyTorch wheels to avoid downloading CUDA packages
RUN pip install --no-cache-dir --retries 10 --timeout 120 \
  --index-url https://download.pytorch.org/whl/cpu \
  torch

WORKDIR /workspace

ENV HF_HOME=/workspace/.hf_cache

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY services /workspace/services

EXPOSE 8000 8001

ENTRYPOINT ["uvicorn"]
CMD ["services.llm_server:app", "--host", "0.0.0.0", "--port", "8000"]
