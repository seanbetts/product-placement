# Use an official PyTorch image as the base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install git and other necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Copy the configuration file and microservice code into the container
COPY sam2_configs/sam2_hiera_l.yaml /app/sam2_configs/
RUN touch /app/sam2_configs/__init__.py
COPY sam2_microservice.py /app/

# Clone the SAM 2 repository
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /app/segment-anything-2

# Install SAM 2
RUN cd /app/segment-anything-2 && pip install -e .

# Install additional dependencies
COPY requirements.txt .
COPY keyfile.json .
RUN pip install --no-cache-dir -r requirements.txt

# Download the SAM 2 model checkpoint
RUN mkdir -p /app/checkpoints
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P /app/checkpoints/

# Set the PYTHONPATH to include the SAM2 directory
ENV PYTHONPATH="${PYTHONPATH}:/app/segment-anything-2"

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "sam2_microservice:app", "--host", "0.0.0.0", "--port", "8080"]