# Use an official PyTorch image as the base image
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone the SAM 2 repository
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /app/segment-anything-2

# Install Python dependencies
RUN pip install --no-cache-dir -e /app/segment-anything-2

# Install additional dependencies for the microservice
RUN pip install --no-cache-dir fastapi uvicorn google-cloud-storage

# Download the SAM 2 model checkpoint
RUN wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P /app/checkpoints/

# Copy the microservice code into the container
COPY sam2_microservice.py /app/

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "sam2_microservice:app", "--host", "0.0.0.0", "--port", "8080"]