FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY data/benchmarks ./data/benchmarks
COPY data/ores ./data/ores
COPY data/templates ./data/templates
COPY data/measured_eps.template.json ./data/measured_eps.template.json
COPY data/lab_measurements.example.json ./data/lab_measurements.example.json

RUN python -m pip install --upgrade pip \
  && pip install -r requirements.txt \
  && pip install -e .

# Default: run unit tests (mirrors CI behavior).
CMD ["python", "-m", "pytest", "-q"]

