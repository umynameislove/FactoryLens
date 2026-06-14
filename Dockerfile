FROM python:3.11-slim@sha256:f9fa7f851e38bfb19c9de3afbc4b86ae7176ea7aaf94535c31df5458d5849457

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home appuser

COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src ./src

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "factorylens.main:app", "--host", "0.0.0.0", "--port", "8000"]
