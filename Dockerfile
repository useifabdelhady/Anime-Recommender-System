FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make sure the artifacts directory exists
RUN mkdir -p artifacts/processed artifacts/weights artifacts/model

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "application.py"]