ARG NODE_IMAGE=node:22-alpine
ARG PYTHON_IMAGE=python:3.11-slim

FROM ${NODE_IMAGE} AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ${PYTHON_IMAGE}
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY . .
RUN pip install --no-cache-dir .
RUN groupadd --system --gid 10001 omnirag \
    && useradd --system --uid 10001 --gid omnirag --no-create-home omnirag \
    && mkdir -p /app/data \
    && chown 10001:10001 /app/data
COPY --from=frontend /ui/dist ./frontend/dist
USER 10001:10001
ENV OMNIRAG_ENVIRONMENT=production OMNIRAG_FRONTEND_DIST=/app/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "omnirag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
