# One image for a laptop, Lambda (via the Lambda Web Adapter) and Kubernetes.
#   docker build --provenance=false -t opportunity-scanner . && docker run -p 8080:8080 opportunity-scanner
#   (--provenance=false: Lambda rejects image indexes with attestations)
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

FROM python:3.13-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY scanner ./scanner
RUN useradd --uid 10001 app
USER app
ENV PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/api/meta PATH=/app/.venv/bin:$PATH
EXPOSE 8080
CMD ["uvicorn", "scanner.app:app", "--host", "0.0.0.0", "--port", "8080"]
