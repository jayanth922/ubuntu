FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p /data/raw /data/processed /data/index

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]