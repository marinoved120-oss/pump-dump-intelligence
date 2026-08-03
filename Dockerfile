FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY research /app/research
COPY orchestrator /app/orchestrator
RUN pip install --upgrade pip && pip install ".[dev]"
COPY configs /app/configs
COPY literature /app/literature
COPY tests /app/tests
COPY PROJECT_CONSTITUTION.yaml ROADMAP.yaml /app/
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts /app/orchestrator_state

ENTRYPOINT ["python", "-m", "research.cli"]
CMD ["--help"]
