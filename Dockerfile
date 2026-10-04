FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements-serve.txt .
# CPU-only torch keeps the image much smaller
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-serve.txt
COPY src ./src
ENV MODEL_DIR=/app/models/v1
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
