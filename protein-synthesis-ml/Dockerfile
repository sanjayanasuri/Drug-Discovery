# Dockerfile for Drug Discovery Pipeline
# Base image with conda and RDKit support
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "hiv-ml", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hiv-ml", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

