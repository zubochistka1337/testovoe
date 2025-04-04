FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt update && \
    apt install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "localhost", "--port", "5000"]