# Build stage: Install required packages and tools, and install Ansible and psycopg2-binary
FROM python:3.10-slim AS builder

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential

# Install Ansible and psycopg2-binary
RUN pip install --no-cache-dir ansible psycopg2-binary

# Final stage: Copy only the necessary files from the build stage
FROM python:3.10-slim

# Install sshpass and postgresql-client in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    sshpass \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the build stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set the working directory
WORKDIR /ansible
