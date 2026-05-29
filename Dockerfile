FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch — keeps the image ~3 GB smaller than the CUDA wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Inference-only Python deps
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Pre-bake the sentence-transformer model into this layer so the container
# starts instantly with no internet access at runtime.
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY predictor_service.py .

# irt_runs/ is mounted at runtime as a read-only volume — not baked in.
# Path must match what predictor_service.py expects:
#   /app/irt_runs/20260525_2339_12000q_12m_4pl_mmlu-boolq-triviaqa/

ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 3001
CMD ["python3", "predictor_service.py", "--host", "0.0.0.0", "--port", "3001"]
