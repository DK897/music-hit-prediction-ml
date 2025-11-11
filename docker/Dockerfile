FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential ffmpeg libsndfile1 && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . /app
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
