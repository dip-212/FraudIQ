# Use a specific version for reproducibility. python:3.12 is a good stable choice.
FROM python:3.12-slim as builder

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy requirements and setup file first to leverage Docker cache
COPY requirements.txt .
COPY setup.py .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.12-slim as final

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

CMD ["python", "application.py"]