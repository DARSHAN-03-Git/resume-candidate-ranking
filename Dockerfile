# Stage 1: Build & export ONNX models (runs once during docker build)
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
	&& pip install --no-cache-dir -r requirements-dev.txt
COPY scripts/export_onnx_models.py ./scripts/
RUN python scripts/export_onnx_models.py --output-root /build/onnx_models

# Stage 2: Lean runtime container (no PyTorch, only onnxruntime + tokenizers)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /build/onnx_models ./onnx_models
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
