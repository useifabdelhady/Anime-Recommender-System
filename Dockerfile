FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make sure the artifacts directory exists
RUN mkdir -p artifacts/processed artifacts/weights artifacts/model

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=application.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]