FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8000

# Default command (overridden by docker-compose)
CMD ["python", "-c", "print('Use docker-compose to run ui + api')"]

