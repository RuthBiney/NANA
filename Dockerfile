# Dockerfile
FROM python:3.8-slim

WORKDIR /mlapp

COPY . /mlapp/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies with a longer timeout
RUN pip install --default-timeout=100 --no-cache-dir -r /mlapp/app/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
