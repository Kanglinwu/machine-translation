# Base image with CUDA 12.1 and cuDNN 8, development environment, Ubuntu 22.04
FROM nvcr.io/nvidia/pytorch:21.11-py3

# Install pip and upgrade setuptools
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && pip install --upgrade setuptools pip

# Install other necessary Python libraries (these should be in your requirements.txt)
RUN pip install uv
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Optional: Remove any unnecessary files to reduce image size
RUN rm -rf /var/lib/apt/lists/* /root/.cache

# Set environment variables to prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the rest of the application code into the container
COPY api.py .
COPY models/ .
COPY conf/ .
COPY docs/ .

# Change the working directory to /app
