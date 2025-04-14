# FROM python:3.11-slim-buster

# WORKDIR /app

# # Create a logs directory
# RUN mkdir -p /app/logs

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Set the log directory as a volume
# VOLUME ["/app/logs"]

# EXPOSE 8000 7860

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
FROM python:3.11-slim-buster

WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Create a logs directory
RUN mkdir -p /app/logs

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

# Set the log directory as a volume
VOLUME ["/app/logs"]

# Expose ports for both FastAPI and Gradio
EXPOSE 8000 7860

# Run the Python script instead of uvicorn directly
CMD ["python", "main.py"]