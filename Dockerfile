# Gunakan image python yang lebih ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy aplikasi
COPY . .

# Buat directory untuk menyimpan uploads dan results
RUN mkdir -p static/uploads static/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Initialize git repository (required by ultralytics)
RUN git init

# Expose port
EXPOSE 5000

# Run aplikasi
CMD ["python", "app.py"]