# The public demo: Streamlit replaying recorded runs through the pipeline.
# The image holds no model SDK and no API key; it never calls a model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# The priorauth package (with its prompt files) and Streamlit.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui]"

# What the demo reads at run time: policies, gold cases, recordings, the app and its settings.
COPY policies ./policies
COPY evals ./evals
COPY ui ./ui
COPY .streamlit ./.streamlit

RUN useradd --create-home demo
USER demo

# Cloud Run sets PORT, 8080 by default.
CMD ["sh", "-c", "exec python -m streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
