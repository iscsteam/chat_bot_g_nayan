FROM python:3.11-slim-buster

WORKDIR /app

# Create a logs directory
RUN mkdir -p /app/logs

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the log directory as a volume
VOLUME ["/app/logs"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]