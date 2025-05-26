# Dockerfile

# 1. Choose a base image
# Using a Python slim image to keep the size down
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependency definition files
# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir reduces image size
# --default-timeout can be increased if downloads are slow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=300

# 5. Copy the rest of your application code into the container
COPY . .
# Ensure .env is NOT copied if it's not in .dockerignore (it's in .gitignore, but explicit .dockerignore is better for build context)

# 6. Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# 7. Define the command to run your application
# Use healthcheck for Streamlit as recommended
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]