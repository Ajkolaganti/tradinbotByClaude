FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Railway injects PORT as env var; default 8080
ENV PORT=8080

EXPOSE 8080

CMD ["python", "main.py"]
