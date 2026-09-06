FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheel keeps the image lean and reproducible.
RUN pip install --no-cache-dir torch>=2.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy>=1.24

COPY . .
ENV PYTHONPATH=/app/src

CMD ["python", "eval/run_eval.py"]
