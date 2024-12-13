# Use Python 3.10 slim-buster as the base image
FROM python:3.10-slim-buster

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libhdf5-dev \
    libgomp1 \
    python-opencv \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libx264-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Set the working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install compatible versions of NumPy and OpenCV first
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Install PyTorch first to satisfy detectron2 dependencies
RUN pip install --no-cache-dir torch==1.11.0 torchvision==0.12.0

# Install detectron2 after PyTorch is installed
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git@v0.6

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models/ /app/models/
COPY detect.py /app/
COPY class_indices.json /app/

# Create outputs directory
RUN mkdir -p /app/outputs

# Expose port
EXPOSE 5005

# Command to run the application
CMD ["python", "detect.py"]
