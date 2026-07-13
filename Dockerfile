# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force PyTorch and Math libraries to respect the 4 vCPU limit
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for compilation (if any)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Add healthcheck to ensure the container is running properly
HEALTHCHECK CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Command to run the application
# Update No-IP DNS securely, then start Streamlit
ENTRYPOINT wget -qO- --http-user="$NOIP_USER" --http-password="$NOIP_PASS" "https://dynupdate.no-ip.com/nic/update?hostname=stockanalyzer.ddns.net" && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false